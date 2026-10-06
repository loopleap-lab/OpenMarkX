"""Tests never touch the real key store: every key they write goes to a throwaway directory."""
import os
import tempfile

os.environ["MWX_KEYS"] = tempfile.mkdtemp(prefix="mwx-test-keys-")
