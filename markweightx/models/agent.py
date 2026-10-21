"""The ReAct agent adapter: a text model run as a small search agent; the output the watermark is checked on is the
agent's final answer. Moved from exp/agent/hotpot_agent.py (behaviour unchanged).

Loop per question (all questions advance together, one batched generation per step):
  assistant: "Thought: ... / Action: Search[title]"  ->  user: "Observation: <that paragraph>"
  ... up to max_steps, until "Action: Finish[answer]"
  user: "Write your final answer for a reader ..."  ->  assistant: the final answer (what the detector sees)
Inputs are {"question": str, "paragraphs": {title: text}}: the search tool's corpus comes in with the data.
"""
from __future__ import annotations

import contextlib
import difflib
import re
import string
from types import SimpleNamespace

import torch

from ..registry import register_model
from .text import TextModel

SYSTEM = ("You answer questions by searching Wikipedia. Work in steps. In each step write one line 'Thought: ...' and "
          "then one line with exactly one action: 'Action: Search[<title>]' to read a Wikipedia paragraph, or "
          "'Action: Finish[<short answer>]' when you know the answer. Titles you can search: {titles}.")
FINAL = ("Now write your final answer for a reader in about {words} words: state the answer, then explain how the "
         "evidence you found supports it. Do not write 'Thought' or 'Action'.")


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def parse_action(text):
    m = re.search(r"Action:\s*(Search|Finish)\[(.*?)\]", text, re.S)
    return (m.group(1), m.group(2).strip()) if m else (None, None)


def search(paragraphs, query):
    titles = list(paragraphs)
    best = difflib.get_close_matches(query, titles, n=1, cutoff=0.0)
    t = best[0] if best else titles[0]
    return f"[{t}] {paragraphs[t]}"


@torch.no_grad()
def generate(model, tok, convs, max_new, temperature, seed, adapter):
    torch.manual_seed(seed)
    chats = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    ctx = model.disable_adapter() if (not adapter and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
    with ctx:
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=1.0,
                           top_k=0, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_agent(model, tok, items, adapter, a, seed):
    convs = [[{"role": "system", "content": SYSTEM.format(titles="; ".join(it["paragraphs"]))},
              {"role": "user", "content": it["question"]}] for it in items]
    finish = [None] * len(items)
    for step in range(a.max_steps):
        live = [i for i in range(len(items)) if finish[i] is None]
        if not live:
            break
        for b in range(0, len(live), a.bs):
            idx = live[b:b + a.bs]
            outs = generate(model, tok, [convs[i] for i in idx], 160, a.temperature, seed * 100 + step * 10 + b, adapter)
            for i, out in zip(idx, outs):
                out = out.split("Observation:")[0].strip()
                kind, arg = parse_action(out)
                if kind:                                    # keep the turn up to its action
                    out = out[:re.search(r"Action:\s*(Search|Finish)\[.*?\]", out, re.S).end()]
                convs[i].append({"role": "assistant", "content": out})
                if kind == "Finish":
                    finish[i] = arg
                elif kind == "Search":
                    convs[i].append({"role": "user", "content": "Observation: " + search(items[i]["paragraphs"], arg)})
                else:
                    convs[i].append({"role": "user", "content": "Observation: no action found; use Search[...] or Finish[...]."})
        print({"step": step + 1, "finished": sum(f is not None for f in finish), "of": len(items)}, flush=True)
    finals = [None] * len(items)
    for b in range(0, len(items), a.bs):
        idx = list(range(b, min(b + a.bs, len(items))))
        for i in idx:
            convs[i].append({"role": "user", "content": FINAL.format(words=a.final_words)})
        outs = generate(model, tok, [convs[i] for i in idx], a.final_tokens, a.temperature, seed * 100 + 99 + b, adapter)
        for i, out in zip(idx, outs):
            finals[i] = out.strip()
            convs[i].append({"role": "assistant", "content": finals[i]})
    return convs, finish, finals



@register_model
class ReActAgent(TextModel):
    name = "agent"
    unit = "token"

    def __init__(self, base: str | None = None, max_steps: int = 6, final_words: int = 300, final_tokens: int = 600,
                 temperature: float = 0.7, bs: int = 50, **params):
        super().__init__(base=base, max_steps=max_steps, final_words=final_words, final_tokens=final_tokens,
                         temperature=temperature, bs=bs, **params)

    def run(self, net, inputs: list, *, seed: int = 0, adapter: bool = True, transcript: bool = False, **kw):
        a = SimpleNamespace(**{k: self.params[k] for k in ("max_steps", "final_words", "final_tokens", "temperature", "bs")})
        convs, finish, finals = run_agent(net, self.tokenizer() if self._tok or self.base else None, inputs, adapter, a, seed)
        self.last_finish = finish
        return (finals, convs) if transcript else finals
