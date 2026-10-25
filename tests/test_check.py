"""openmarkx check must fail each broken adapter in tests/broken/ on the check that targets its defect."""
import os

import pytest

from openmarkx.check import run

BROKEN = os.path.join(os.path.dirname(__file__), "broken")
CASES = [  # file, scheme name, the check that must fail, the defect
    ("rev_p.py", "rev-p", "C5", "p-value from the wrong tail"),
    ("tail10.py", "tail10", "C5", "p-value shrunk tenfold below 0.05"),
    ("badp.py", "bad-badp", "C5", "p-value far too small"),
    ("ctx.py", "ctx", "C10", "training Signal differs from the decoding Signal"),
    ("leak_edit.py", "leak-edit", "C7", "secret written into the release"),
    ("crash.py", "bad-crash", "C4", "score() crashes"),
    ("leak.py", "bad-leak", "C4", "declares R0 but runs the model"),
    ("nokey.py", "bad-nokey", "C4", "ignores the key"),
    ("nosig.py", "bad-nosig", "C6", "no Signal"),
    ("unseeded.py", "bad-unseeded", "C3", "unseeded randomness"),
    ("capn.py", "capn", "C5", "token count capped: p-values wrong on long text"),
    ("rep.py", "rep", "C5", "repeated tokens counted: p-values wrong on looping text"),
    ("cfgleak.py", "cfgleak", "C7", "secret written into config.json"),
    ("hashrand.py", "hashrand", "C3|C4", "depends on Python's hash() seed (its key space also collides)"),
    ("lowkey.py", "lowkey", "C4", "only 16 effective keys"),
    ("mutmodel.py", "mutmodel", "C8", "setup() changes the shared tokenizer"),
    ("multi/wm.py", "multi", "C3", "imports a sibling file the key does not carry"),
]


@pytest.mark.slow
@pytest.mark.parametrize("file,name,must_fail,defect", CASES, ids=[c[0] for c in CASES])
def test_check_catches(file, name, must_fail, defect, monkeypatch):
    monkeypatch.syspath_prepend(os.path.dirname(os.path.join(BROKEN, file)))   # as when run next to the file
    rows = run(name, source=os.path.join(BROKEN, file), n_null=50)
    failed = {r[0].split()[0] for r in rows if r[1] == "FAIL"}
    assert failed & set(must_fail.split("|")), f"{defect}: {must_fail} passed\n" + "\n".join(map(str, rows))
