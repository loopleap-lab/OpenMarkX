"""Where things live on this server. Override with environment variables."""
import os

RUNS = os.environ.get("MWX_RUNS", "/mnt/data/jiaq/mwx-runs")
MARKLLM = os.environ.get("MARKLLM_DIR", "/mnt/data/jiaq/MarkLLM")          # upstream MarkLLM @112fa74, unchanged
PRO_CODE = os.environ.get("PRO_CODE_DIR", "/home/jiaq/Research/PRO/refs/PRO-code")   # PRO paper code, unchanged
HF = os.environ.get("MWX_HF", "/mnt/data/jiaq/hf")
BASES = {
    "llama-3.2-1b": f"{HF}/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    "llama-3.1-8b-instruct": f"{HF}/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/{{}}",
    "llama-3.2-1b-instruct": f"{HF}/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "llama-3.1-8b": next(p for p in (
        "/mnt/data/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",   # lab2
        f"{HF}/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",            # lab0 copy
    ) if os.path.isdir(p) or p.startswith(HF)),
}
# both bases share the Llama-3 tokenizer (128256), so one tokenized corpus serves both
DATA = {"llama3": os.environ.get("MWX_DATA", f"{RUNS}/data_llama8b_256.pt")}   # made by exp/prepare_data.py
HUB = {  # where a base comes from when its pinned snapshot is not on disk (downloaded on first use)
    "llama-3.2-1b": "unsloth/Llama-3.2-1B", "llama-3.2-1b-instruct": "unsloth/Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B", "llama-3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
}


def base_path(name: str) -> str:
    p = BASES.get(name, name)
    if p.endswith("/{}"):                                   # snapshot hash not pinned yet: take the one on disk
        root = p[:-3]
        p = os.path.join(root, sorted(os.listdir(root))[0]) if os.path.isdir(root) else root
    if name in HUB and not os.path.isdir(p):               # not on this machine: the Hugging Face repo
        return HUB[name]
    return p
