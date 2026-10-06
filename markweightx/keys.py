"""Key store: keys live apart from releases, filed by commitment. A release (and every edit of it) carries only the
commitment, so whoever holds the store can find the key, and whoever holds the release cannot.

Files that are part of a key (PRO's token->dimension mapping, a trained policy) are stored content-addressed under
blobs/ and referenced from params as "blob:<sha256>", so a commitment never depends on where a file lives."""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil

from .paths import RUNS
from .types import Key, Release

STORE = os.environ.get("MWX_KEYS", os.path.join(RUNS, "keys"))


def new_secret(bits: int = 40) -> int:
    """A fresh random secret, small enough for every teacher's seeding arithmetic (seed * token id < 2**64)."""
    return secrets.randbits(bits)


def key_path(commitment: str) -> str:
    return os.path.join(STORE, f"{commitment}.json")


def put_blob(path: str) -> str:
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ext = os.path.splitext(path)[1]
    dst = os.path.join(STORE, "blobs", h + ext)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(path, dst)
    return f"blob:{h}{ext}"


def resolve(ref):
    """'blob:<sha256>.ext' -> its path in the store; anything else unchanged."""
    if isinstance(ref, str) and ref.startswith("blob:"):
        return os.path.join(STORE, "blobs", ref[5:])
    return ref


def store(key: Key) -> str:
    """Move file-valued params into the blob store, then file the key under its commitment."""
    for k, v in list(key.params.items()):
        if isinstance(v, str) and not v.startswith("blob:") and os.path.isfile(v):
            key.params[k] = put_blob(v)
    return key.save(key_path(key.commitment))


def load_key(release: Release | str) -> Key:
    """The key of a release: from the store by commitment, or the legacy copy inside release.json."""
    rel = Release.open(release) if isinstance(release, str) else release
    if rel.key_commitment and os.path.exists(key_path(rel.key_commitment)):
        return Key.open(key_path(rel.key_commitment))
    if rel.key:
        return Key(scheme=rel.scheme, params=dict(rel.key), base=rel.base)
    raise FileNotFoundError(f"no key for release {rel.path}: commitment {rel.key_commitment} not in {STORE}")
