"""Key store: keys live apart from releases, filed by commitment. A release (and every edit of it) carries only the
commitment, so whoever holds the store can find the key, and whoever holds the release cannot."""
from __future__ import annotations

import os

from .paths import RUNS
from .types import Key, Release

STORE = os.environ.get("MWX_KEYS", os.path.join(RUNS, "keys"))


def key_path(commitment: str) -> str:
    return os.path.join(STORE, f"{commitment}.json")


def store(key: Key) -> str:
    return key.save(key_path(key.commitment))


def load_key(release: Release | str) -> Key:
    """The key of a release: from the store by commitment, or the legacy copy inside release.json."""
    rel = Release.open(release) if isinstance(release, str) else release
    if rel.key_commitment and os.path.exists(key_path(rel.key_commitment)):
        return Key.open(key_path(rel.key_commitment))
    if rel.key:
        return Key(scheme=rel.scheme, params=dict(rel.key))
    raise FileNotFoundError(f"no key for release {rel.path}: commitment {rel.key_commitment} not in {STORE}")
