"""CPU: the image+question data for sub-project B, under $OMX_RUNS/med/.

  python exp/med/prepare.py
mxqa_mm_test.jsonl  MedXpertQA-MM test (2000 questions; images unzipped to mxqa_images/), the evaluation set
mxqa_mm_train.jsonl / mxqa_mm_eval.jsonl
                    the 2000 questions split in half (seed 0, stratified by medical task): the distillation prompts
                    come from train (questions only; answers are the model's own), the numbers from eval
vqa_prompts.jsonl   (earlier design, unused now) image+question prompts from the training splits of VQA-RAD (radiology) and PathVQA (pathology),
                    for the distillation data; never from MedXpertQA
"""
import argparse
import hashlib
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openmarkx.paths import RUNS

MED = os.path.join(RUNS, "med")
EVAL_PROMPT = ("{question}\n\nThink step by step about the image(s) and the question, then finish with "
               "\"The answer is (X)\", where X is the letter of your choice.")
DISTILL_PROMPT = ("Describe the relevant findings in the image, reason step by step, and then answer the question.\n"
                  "Question: {question}")


def mxqa():
    from huggingface_hub import snapshot_download
    root = snapshot_download("TsinghuaC3I/MedXpertQA", repo_type="dataset")
    img_dir = os.path.join(MED, "mxqa_images")
    if not os.path.isdir(img_dir):
        with zipfile.ZipFile(os.path.join(root, "images.zip")) as z:
            z.extractall(img_dir + ".tmp")
        os.rename(img_dir + ".tmp", img_dir)
    found = {f: os.path.join(d, f) for d, _, fs in os.walk(img_dir) for f in fs}
    n, missing = 0, 0
    with open(os.path.join(MED, "mxqa_mm_test.jsonl"), "w") as out:
        for line in open(os.path.join(root, "MM", "test.jsonl")):
            r = json.loads(line)
            imgs = [found.get(x) for x in r["images"]]
            if None in imgs:
                missing += 1
                continue
            out.write(json.dumps({"id": r["id"], "images": imgs, "text": EVAL_PROMPT.format(question=r["question"]),
                                  "label": r["label"], "medical_task": r.get("medical_task"),
                                  "body_system": r.get("body_system"), "question_type": r.get("question_type")}) + "\n")
            n += 1
    print({"mxqa_mm_test": n, "missing_images": missing})


def split(seed=0):
    """Half of each medical task to train, half to eval, by question id (never the same question in both)."""
    rows = [json.loads(l) for l in open(os.path.join(MED, "mxqa_mm_test.jsonl"))]
    by_task = {}
    for r in rows:
        by_task.setdefault(r.get("medical_task"), []).append(r)
    rng, train, ev = random.Random(seed), [], []
    for task in sorted(by_task, key=str):
        group = sorted(by_task[task], key=lambda r: r["id"])
        rng.shuffle(group)
        train += group[:len(group) // 2]; ev += group[len(group) // 2:]
    for name, part in (("train", train), ("eval", ev)):
        part.sort(key=lambda r: int(r["id"].split("-")[1]))
        with open(os.path.join(MED, f"mxqa_mm_{name}.jsonl"), "w") as out:
            out.writelines(json.dumps(r) + "\n" for r in part)
    print({"train": len(train), "eval": len(ev)})


def vqa(n_total, seed):
    from datasets import load_dataset
    img_dir = os.path.join(MED, "vqa_images"); os.makedirs(img_dir, exist_ok=True)
    rows, seen = [], set()
    for name in ("flaviagiammarino/vqa-rad", "flaviagiammarino/path-vqa"):
        for ex in load_dataset(name, split="train"):
            img = ex["image"].convert("RGB")
            h = hashlib.sha1(img.tobytes()).hexdigest()[:16]
            path = os.path.join(img_dir, f"{h}.png")
            if not os.path.exists(path):
                img.save(path)
            if (h, ex["question"]) in seen:
                continue
            seen.add((h, ex["question"]))
            rows.append({"id": f"{name.split('/')[1]}-{len(rows)}", "images": [path],
                         "text": DISTILL_PROMPT.format(question=ex["question"]), "source": name})
        print({name: len(rows)}, flush=True)
    rad = [r for r in rows if "vqa-rad" in r["source"]]
    path_ = [r for r in rows if "path-vqa" in r["source"]]
    random.Random(seed).shuffle(path_)
    keep = rad + path_[:max(0, n_total - len(rad))]
    random.Random(seed).shuffle(keep)
    with open(os.path.join(MED, "vqa_prompts.jsonl"), "w") as out:
        for r in keep:
            out.write(json.dumps(r) + "\n")
    print({"vqa_prompts": len(keep), "vqa-rad": len(rad), "path-vqa_used": len(keep) - len(rad)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_vqa", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vqa", action="store_true", help="also the VQA-RAD/PathVQA prompts (earlier design)")
    a = ap.parse_args()
    os.makedirs(MED, exist_ok=True)
    mxqa()
    split(a.seed)
    if a.vqa:
        vqa(a.n_vqa, a.seed)
