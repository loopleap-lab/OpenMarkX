import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))
import yaml

from markweightx.registry import SCHEMES, get_scheme

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_finished_scheme_has_a_matching_yaml():
    from write_configs import row
    get_scheme("nowm")
    names = [n for n, c in SCHEMES.items() if not getattr(c, "experimental", False)]
    for n in names:
        f = os.path.join(ROOT, "configs", "watermark", f"{n}.yaml")
        assert os.path.exists(f), f"missing {f}: run exp/write_configs.py"
        assert yaml.safe_load(open(f)) == row(n), f"{f} is out of date: run exp/write_configs.py"


def test_cli_new_scheme_then_check(tmp_path):
    import subprocess
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="")
    f = tmp_path / "toy_scheme.py"
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "new-scheme", "toy-scheme", "--out", str(f)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "check", "--adapter", str(f)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "FAIL" not in r.stdout, r.stdout + r.stderr
    assert "C3 round trip   PASS" in r.stdout
