"""Key store: keys live apart from releases, filed by commitment. A release (and every edit of it) carries only the
commitment, so whoever holds the store can find the key, and whoever holds the release cannot.

Files that are part of a key (PRO's token->dimension mapping, a trained policy) are stored content-addressed under
blobs/ and referenced from params as "blob:<sha256>", so a commitment never depends on where a file lives."""
from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import shutil

from .paths import RUNS
from .types import Key, Release

STORE = os.environ.get("MWX_KEYS", os.path.join(RUNS, "keys"))
PUBLIC_KEY_FIELDS = {"rng_device", "dtype", "base_sha256", "public"}    # key fields that are not secret (shown by `keys show`)


def new_secret(bits: int = 40) -> int:
    """A fresh random secret, small enough for every teacher's seeding arithmetic (seed * token id < 2**64)."""
    return secrets.randbits(bits)


def key_path(commitment: str) -> str:
    return os.path.join(STORE, f"{commitment}.json")


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def put_blob(path: str) -> str:
    """Copy a key file into the store under its sha256. Written to a temporary name and renamed, and an existing blob
    is checked: a truncated copy (full disk) never stands in for the real one."""
    h = _sha(path)
    ext = os.path.splitext(path)[1]
    dst = os.path.join(STORE, "blobs", h + ext)
    if not os.path.exists(dst) or _sha(dst) != h:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.tmp{os.getpid()}"
        shutil.copyfile(path, tmp)
        if _sha(tmp) != h:
            os.remove(tmp)
            raise OSError(f"copying {path} into the key store did not reproduce it (disk full?)")
        os.replace(tmp, dst)
    return f"blob:{h}{ext}"


def resolve(ref):
    """'blob:<sha256>.ext' -> its path in the store; anything else unchanged."""
    if isinstance(ref, str) and ref.startswith("blob:"):
        return os.path.join(STORE, "blobs", ref[5:])
    return ref


def stage(name: str) -> str:
    """A private path to write key material to before store() files it (e.g. a mapping, a trained policy, a
    perturbation). Never inside a release directory."""
    d = os.path.join(STORE, "tmp")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{os.urandom(8).hex()}-{name}")


def stage_dir() -> str:
    """A fresh private directory for a scheme's save(): key material written there is filed by store()."""
    d = os.path.join(STORE, "tmp", os.urandom(8).hex())
    os.makedirs(d)
    return d


def store(key: Key) -> str:
    """Move file-valued params (and a plugin scheme's source file) into the blob store, then file the key under its
    commitment. Keys are the owner's own artifacts: a key's plugin source is code that detection imports."""
    if key.scheme_source and key.scheme_source.endswith(".py") and os.path.isfile(key.scheme_source):
        key.scheme_source = put_blob(key.scheme_source)
    tmp = os.path.join(STORE, "tmp") + os.sep
    for k, v in list(key.params.items()):
        if isinstance(v, str) and os.path.abspath(v).startswith(tmp) and not os.path.isfile(v):
            raise FileNotFoundError(f"key field {k}: the staged file {v} is gone; the key cannot be stored")
        if isinstance(v, str) and not v.startswith("blob:") and os.path.isfile(v):
            key.params[k] = put_blob(v)
            if os.path.abspath(v).startswith(tmp):        # staged key material: the blob is now the only copy
                os.remove(v)
                if os.path.dirname(os.path.abspath(v)) != tmp.rstrip(os.sep):
                    try:
                        os.rmdir(os.path.dirname(v))        # an emptied stage_dir()
                    except OSError:
                        pass
    return key.save(key_path(key.commitment))


def load_key(release: Release | str) -> Key:
    """The key of a release: from the store by commitment, or the legacy copy inside release.json."""
    rel = Release.open(release) if isinstance(release, str) else release
    if rel.key_commitment and os.path.exists(key_path(rel.key_commitment)):
        return Key.open(key_path(rel.key_commitment))
    if rel.key:
        import warnings
        warnings.warn(f"{rel.path}: using the key copy inside release.json (old format). A release can carry any key; "
                      f"use a key from your own key store when it matters who made it.")
        return Key(scheme=rel.scheme, params=dict(rel.key), base=rel.base)
    raise FileNotFoundError(f"no key for release {rel.path}: commitment {rel.key_commitment} not in {STORE}")
