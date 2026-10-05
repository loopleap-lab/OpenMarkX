"""Where things live on this server. Override with environment variables."""
import os

RUNS = os.environ.get("MWX_RUNS", "/mnt/data/jiaq/mwx-runs")
MARKLLM = os.environ.get("MARKLLM_DIR", "/mnt/data/jiaq/MarkLLM")          # upstream MarkLLM @112fa74, unchanged
PRO_CODE = os.environ.get("PRO_CODE_DIR", "/home/jiaq/Research/PRO/refs/PRO-code")   # PRO paper code, unchanged
BASES = {
    "llama-3.2-1b": "/mnt/data/jiaq/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    "llama-3.1-8b": "/mnt/data/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
}
# both bases share the Llama-3 tokenizer (128256), so one tokenized corpus serves both
DATA = {"llama3": f"{RUNS}/data_llama8b_256.pt"}


def base_path(name: str) -> str:
    return BASES.get(name, name)
