"""GPU: the distillation corpus for an instruct model: its own answers to Dolly instructions, in chat format.

  python exp/agent/distill_data.py --base llama-3.1-8b-instruct --n 9000 --shard 0 --num_shards 1
Each training chunk is the last --seq_len tokens of (chat prompt + the model's answer), the answer forced to
--answer_len tokens, so every chunk contains the assistant header and mostly answer text. Writes the corpus format of
exp/prepare_data.py ({"train": [N, seq_len] tokens, "eval_texts": held-out answers}) to
$OMX_RUNS/agent/data_<base>.pt; shards go to .s<k>of<N>.partial.jsonl first (resumable), then a merge step:
  python exp/agent/distill_data.py --merge --num_shards 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from openmarkx.paths import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--answer_len", type=int, default=224)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=600, help="held-out answers (the last ones) for evaluation prompts")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_prompt", type=int, default=768, help="skip prompts longer than this (a few Dolly contexts are "
                    "thousands of tokens; the first generation step holds logits for every prompt position)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    d = os.path.join(RUNS, "agent")
    part = lambda k: os.path.join(d, f"distill_{a.base}.s{k}of{a.num_shards}.partial.jsonl")
    if a.merge:
        rows = [json.loads(l) for k in range(a.num_shards) for l in open(part(k))]
        rows.sort(key=lambda r: r["i"])
        train = [r["chunk"] for r in rows[:-a.n_eval] if len(r["chunk"]) == a.seq_len]
        evals = [r["answer"] for r in rows[-a.n_eval:]]
        out = os.path.join(d, f"data_{a.base}.pt")
        torch.save({"train": torch.tensor(train), "eval_texts": evals, "model": a.base, "source": "dolly-self"}, out)
        print({"train": len(train), "eval": len(evals), "out": out}); return
    from openmarkx.models import load_base, load_tokenizer
    tok = load_tokenizer(a.base)
    prompts = [json.loads(l)["text"] for l in open(os.path.join(d, "dolly_prompts.jsonl"))][:a.n]
    mine = list(range(a.shard, len(prompts), a.num_shards))
    done = {json.loads(l)["i"] for l in open(part(a.shard))} if os.path.exists(part(a.shard)) else set()
    mine = [i for i in mine if i not in done]
    chat_len = lambda i: len(tok.apply_chat_template([{"role": "user", "content": prompts[i]}], add_generation_prompt=True))
    mine = [i for i in mine if chat_len(i) <= a.max_prompt]
    model = load_base(a.base, "cuda:0")
    with open(part(a.shard), "a") as f, torch.no_grad():
        for b in range(0, len(mine), a.bs):
            idx = mine[b:b + a.bs]
            chats = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                             add_generation_prompt=True) for i in idx]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            torch.manual_seed(1000 + idx[0])
            o = model.generate(**enc, max_new_tokens=a.answer_len, min_new_tokens=a.answer_len, do_sample=True,
                               temperature=1.0, top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                p_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()
                ans = o[j, enc["input_ids"].shape[1]:].tolist()
                f.write(json.dumps({"i": i, "chunk": (p_ids + ans)[-a.seq_len:],
                                    "answer": tok.decode(ans, skip_special_tokens=True)}) + "\n")
            f.flush()
            print({"done": b + len(idx), "of": len(mine)}, flush=True)


if __name__ == "__main__":
    main()
