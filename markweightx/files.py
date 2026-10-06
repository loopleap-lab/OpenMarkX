"""Small file helpers."""
import json
import os


def write_json(path: str, obj, **kw) -> None:
    """Write JSON through a temporary file and a rename, so a full disk or a crash never leaves half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)
