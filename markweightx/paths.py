"""Where things live on this server. Override with environment variables."""
import os

RUNS = os.environ.get("MWX_RUNS", "/mnt/data/jiaq/mwx-runs")
MARKLLM = os.environ.get("MARKLLM_DIR", "/mnt/data/jiaq/MarkLLM")          # upstream MarkLLM @112fa74, unchanged
PRO_CODE = os.environ.get("PRO_CODE_DIR", "/home/jiaq/Research/PRO/refs/PRO-code")   # PRO paper code, unchanged
HF = os.environ.get("MWX_HF", "/mnt/data/jiaq/hf")
BASES = {
    "llama-3.2-1b": f"{HF}/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    "llama-3.2-1b-instruct": f"{HF}/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "llama-3.1-8b": next(p for p in (
        "/mnt/data/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",   # lab2
        f"{HF}/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",            # lab0 copy
    ) if os.path.isdir(p) or p.startswith(HF)),
}
# both bases share the Llama-3 tokenizer (128256), so one tokenized corpus serves both
DATA = {"llama3": f"{RUNS}/data_llama8b_256.pt"}


def base_path(name: str) -> str:
    return BASES.get(name, name)
