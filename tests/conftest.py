"""Tests never touch the real key store: every key they write goes to a throwaway directory."""
import os
import tempfile

for _k in [k for k in os.environ if k.startswith("MWX_")]:   # old names would stand in for unset OMX_ ones;
    del os.environ[_k]                                        # tests of that fallback set them themselves
os.environ["OMX_KEYS"] = tempfile.mkdtemp(prefix="omx-test-keys-")
