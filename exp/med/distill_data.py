"""GPU: the base model's own answers to the training half of MedXpertQA-MM, the distillation set for embed.

  python exp/med/distill_data.py --base medgemma-1.5-4b-it --samples 8 --answer_len 256
Each training question is answered --samples times (different seeds); only the questions are used, never the labels.
Writes $MWX_RUNS/med/pairs_<base>_mxqa.jsonl: {"id", "images", "text", "answer_ids"}, each answer exactly --answer_len
tokens (sampled at temperature 1 with EOS held off until then, as the text-only corpus is cut into fixed chunks).
Resumable: finished batches are kept in <out>.partial.jsonl and a rerun continues after them.
Shards: --shard k --num_shards N answers every N-th item from k and writes <out stem>.s<k>of<N>.jsonl (one GPU each,
e.g. a Slurm array); embed reads all shards.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from markweightx.paths import RUNS
from markweightx.vlm import generate_vlm, load_processor, load_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="medgemma-1.5-4b-it")
    ap.add_argument("--prompts", default=os.path.join(RUNS, "med", "mxqa_mm_train.jsonl"))
    ap.add_argument("--n", type=int, default=None, help="use only the first n questions")
    ap.add_argument("--samples", type=int, default=8, help="answers per question")
    ap.add_argument("--answer_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    out = a.out or os.path.join(RUNS, "med", f"pairs_{a.base}_mxqa.jsonl")
    if a.num_shards > 1:
        out = out.replace(".jsonl", f".s{a.shard}of{a.num_shards}.jsonl")
    if os.path.exists(out):
        print({"pairs": out, "note": "already there; nothing to do"})
        return
    qs = [json.loads(l) for l in open(a.prompts)][:a.n]
    items = [{"id": f"{q['id']}#{k}", "images": q["images"], "text": q["text"]} for k in range(a.samples) for q in qs]
    items = items[a.shard::a.num_shards]
    proc, model = load_processor(a.base), load_vlm(a.base, a.device)
    ids = generate_vlm(model, proc, items, new_tokens=a.answer_len, min_new_tokens=a.answer_len, bs=a.bs, seed=7 + 1000 * a.shard,
                       return_ids=True, partial=out + ".partial.jsonl")
    from markweightx.files import write_text
    write_text(out, "".join(json.dumps({**it, "answer_ids": x}) + "\n" for it, x in zip(items, ids)))
    os.remove(out + ".partial.jsonl")
    full = sum(len(x) >= a.answer_len for x in ids)
    print({"pairs": len(ids), "questions": len(qs), "full_length": full, "out": out})
    print("example:", proc.tokenizer.decode(ids[0]))


if __name__ == "__main__":
    main()
