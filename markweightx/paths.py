"""Where things live. Every location can be set by an environment variable; the defaults are under ~/.markweightx.

  MWX_RUNS       releases, results, the key store and the tokenized corpus     (~/.markweightx/runs)
  MWX_KEYS       the key store                                                    ($MWX_RUNS/keys)
  MWX_DATA       the tokenized C4 corpus made by exp/prepare_data.py             ($MWX_RUNS/data_llama8b_256.pt)
  MARKLLM_DIR    a checkout of upstream MarkLLM at commit 112fa74               (~/.markweightx/MarkLLM)
  PRO_CODE_DIR   the PRO paper code                                              (~/.markweightx/PRO-code)
  HF_HOME        the Hugging Face cache that base models are downloaded into   (Hugging Face's default)
"""
import os

HOME = os.path.expanduser(os.environ.get("MWX_HOME", "~/.markweightx"))
RUNS = os.environ.get("MWX_RUNS", os.path.join(HOME, "runs"))
MARKLLM = os.environ.get("MARKLLM_DIR", os.path.join(HOME, "MarkLLM"))    # upstream MarkLLM @112fa74, unchanged
PRO_CODE = os.environ.get("PRO_CODE_DIR", os.path.join(HOME, "PRO-code"))  # PRO paper code, unchanged
MARKLLM_COMMIT = "112fa74"

# Base model names -> (Hugging Face repo, pinned revision). A pinned revision makes runs comparable across machines;
# None takes the repo's current revision.
BASES = {
    "llama-3.2-1b": ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da"),
    "llama-3.2-1b-instruct": ("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "llama-3.1-8b-instruct": ("unsloth/Meta-Llama-3.1-8B-Instruct", None),
}
# both bases share the Llama-3 tokenizer (128256), so one tokenized corpus serves both
DATA = {"llama3": os.environ.get("MWX_DATA", os.path.join(RUNS, "data_llama8b_256.pt"))}


def base_path(name: str) -> str:
    """A local directory for a base model: a known name is fetched at its pinned revision into the Hugging Face cache
    (once; later calls use the cache), a local directory is used as is, anything else is taken as a repo id."""
    if os.path.isdir(name):
        return name
    if name not in BASES:
        return name
    from huggingface_hub import snapshot_download
    repo, rev = BASES[name]
    try:
        return snapshot_download(repo, revision=rev, local_files_only=True)
    except Exception:                                       # not cached yet (or a partial download): fetch it
        return snapshot_download(repo, revision=rev)
