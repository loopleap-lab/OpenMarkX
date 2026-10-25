"""Small file helpers."""
import json
import os


def write_json(path: str, obj, **kw) -> None:
    """Write JSON through a temporary file and a rename, so a full disk or a crash never leaves half a file."""
    write_text(path, json.dumps(obj, **kw))


def write_text(path: str, text: str) -> None:
    """The same for text: written to a temporary file, then renamed over `path` (removed if writing fails)."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
