"""Where things live. Every location can be set by an environment variable; the defaults are under ~/.openmarkx.

  OMX_HOME       the default root of everything below                           (~/.openmarkx)
  OMX_RUNS       releases, results, the key store and the tokenized corpus     ($OMX_HOME/runs)
  OMX_KEYS       the key store                                                    ($OMX_RUNS/keys)
  OMX_DATA       the tokenized C4 corpus made by exp/prepare_data.py (a file)    ($OMX_RUNS/data_llama8b_256.pt)
  OMX_DEMO_DATA  the directory the demo configs read their inputs from           ($OMX_RUNS)
  MARKLLM_DIR    a checkout of upstream MarkLLM at commit 112fa74               ($OMX_HOME/MarkLLM)
  HF_HOME        the Hugging Face cache that base models are downloaded into   (Hugging Face's default)

Names from before the rename to OpenMarkX still work: each OMX_* variable falls back to its MWX_* namesake, and when
~/.openmarkx does not exist but ~/.markweightx does, the old directory is used.
"""
import os

OLD_PREFIX, PREFIX = "MWX_", "OMX_"


def env(name: str, default=None):
    """$OMX_<name>, else the old $MWX_<name>, else default."""
    v = os.environ.get(PREFIX + name)
    return v if v is not None else os.environ.get(OLD_PREFIX + name, default)


def _default_home() -> str:
    new, old = os.path.expanduser("~/.openmarkx"), os.path.expanduser("~/.markweightx")
    return old if not os.path.exists(new) and os.path.exists(old) else new


HOME = os.path.expanduser(env("HOME") or _default_home())
RUNS = env("RUNS", os.path.join(HOME, "runs"))
MARKLLM = os.environ.get("MARKLLM_DIR", os.path.join(HOME, "MarkLLM"))    # upstream MarkLLM @112fa74, unchanged
MARKLLM_COMMIT = "112fa74"

# Base model names -> (Hugging Face repo, pinned revision). A pinned revision makes runs comparable across machines;
# None takes the repo's current revision.
BASES = {
    "llama-3.2-1b": ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da"),
    "llama-3.2-1b-instruct": ("unsloth/Llama-3.2-1B-Instruct", "5a8abab4a5d6f164389b1079fb721cfab8d7126c"),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "llama-3.1-8b-instruct": ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1"),
    # image+text input, text output (openmarkx.vlm); gated: accept the terms on Hugging Face first
    "medgemma-1.5-4b-it": ("google/medgemma-1.5-4b-it", "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"),
}
# both bases share the Llama-3 tokenizer (128256), so one tokenized corpus serves both
DATA = {"llama3": env("DATA", os.path.join(RUNS, "data_llama8b_256.pt"))}


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
