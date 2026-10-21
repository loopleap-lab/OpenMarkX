"""markweightx run: one pipeline for every adapter, end to end on CPU stand-ins; resume; config validation; the CLI."""
import json
import math
import os
import subprocess
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)
sys.path.insert(0, ROOT)
import pytest
import torch
import yaml

TOY = '''
import math
import torch
import markweightx as mwx


@mwx.register_scheme
class EvenToy(mwx.Scheme):
    """A toy green list: even token ids are green."""
    name = "eventoy"
    exact_null = True

    def __init__(self, delta: float = 4.0):
        super().__init__(delta=delta)

    def target_logits(self, input_ids, logits):
        add = torch.zeros(logits.shape[-1], device=logits.device)
        add[::2] = self.params["delta"]
        return logits.float() + add

    def score(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        g = sum(1 for i in ids if i % 2 == 0)
        return ((g - 0.5 * n) / math.sqrt(max(n, 1) * 0.25), n)

    def pvalue(self, statistic, n_tokens):
        return 0.5 * math.erfc(statistic / math.sqrt(2))
'''

CHAT = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
REPORT_KEYS = {"model", "scheme", "detect", "quality", "edits", "settings"}


@pytest.fixture(scope="module")
def text_base(tmp_path_factory):
    """A tiny random Llama saved as a local base directory, with the real Llama-3 tokenizer and a chat template."""
    from fakes import tiny_llama
    from markweightx.models import load_tokenizer
    d = tmp_path_factory.mktemp("base")
    tiny_llama().save_pretrained(d)
    tok = load_tokenizer("llama-3.2-1b")
    tok.chat_template = CHAT
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    f = tmp_path / "eventoy.py"
    f.write_text(TOY)
    return str(f)


def write_cfg(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def check_report(out):
    rep = json.load(open(os.path.join(out, "report.json")))
    assert REPORT_KEYS <= set(rep)
    assert {"tpr_at_1fpr", "thr", "auc", "n_wm", "n_null"} <= set(rep["detect"])
    assert {"before", "after"} <= set(rep["quality"])
    assert finite(rep["detect"]) and finite(rep["quality"]) and finite(rep["edits"])
    v = json.load(open(os.path.join(out, "assets", "viewer.json")))   # what the run viewer on the site reads
    assert {"model", "scheme", "steps", "detect", "quality", "example"} <= set(v)
    assert [s["step"] for s in v["steps"]][:3] == ["prepare", "selfgen", "calibrate"] and v["detect"]["wm"]
    EXAMPLE = {"text": {"prompt", "marked", "base"}, "agent": {"question", "marked", "base"},
               "vlm": {"question", "image", "marked", "base"}, "taming": {"marked_png", "base_png", "marked_green", "base_green"},
               "geopt": {"x", "y", "field", "pattern", "green_key", "green_wrong"}}[rep["model"]["name"]]
    assert EXAMPLE <= set(v["example"]), set(v["example"])
    for k in ("image", "marked_png", "base_png"):
        if k in v["example"]:
            assert os.path.isfile(os.path.join(out, v["example"][k])), k
    return rep


def pipeline_steps():
    from markweightx.pipeline import STEPS
    return STEPS


def text_cfg(base, toy, out, **over):
    cfg = {"model": {"name": "text", "base": base},
           "scheme": {"name": "eventoy", "source": toy, "params": {"delta": 4.0}},
           "data": {"name": "synthetic"},
           "imprint": {"phases": [{"steps": 3, "lr": 1e-3, "rank": 4}], "bs": 2, "seq_len": 16},
           "generate": {"n": 4, "new_tokens": 12},
           "detect": {"fpr": 0.01, "null": "base"},
           "edits": ["prune:sparsity=0.1"],
           "quality": "ppl",
           "out": str(out)}
    cfg.update(over)
    return cfg


def run_cfg(cfg, tmp_path, **kw):
    from markweightx import pipeline
    return pipeline.run(pipeline.load_config(write_cfg(tmp_path, cfg)), device="cpu", **kw)


def test_run_end_to_end_text(tmp_path, text_base, toy):
    run_cfg(text_cfg(text_base, toy, tmp_path / "run"), tmp_path)
    rep = check_report(tmp_path / "run")
    assert "prune:sparsity=0.1" in rep["edits"]
    prog = [json.loads(l) for l in open(tmp_path / "run" / "progress.jsonl")]
    assert [r["step"] for r in prog if r["event"] == "done"] == list(__import__("markweightx.pipeline").pipeline.STEPS)
    assert pipeline_steps()[-1] == "viewer"


def test_run_end_to_end_agent(tmp_path, text_base, toy):
    cfg = text_cfg(text_base, toy, tmp_path / "run", edits=[], quality="accuracy", generate={"n": 4},
                   model={"name": "agent", "base": text_base, "max_steps": 2, "final_words": 10, "final_tokens": 10})
    run_cfg(cfg, tmp_path)
    check_report(tmp_path / "run")


def test_run_end_to_end_vlm(tmp_path, toy):
    cfg = {"model": {"name": "vlm", "base": "tiny", "factory": "fakes:tiny_gemma", "processor": "fakes:FakeProcessor"},
           "scheme": {"name": "eventoy", "source": toy},
           "data": {"name": "synthetic"},
           "imprint": {"phases": [{"steps": 3, "lr": 1e-3, "rank": 4}], "bs": 2, "answer_len": 4},
           "generate": {"n": 4, "new_tokens": 6},
           "detect": {"null": "base"}, "edits": [], "quality": "accuracy", "out": str(tmp_path / "run")}
    run_cfg(cfg, tmp_path)
    check_report(tmp_path / "run")


def test_run_end_to_end_taming(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    cfg = {"model": {"name": "taming", "factory": "fakes:FakeTaming"},
           "scheme": {"name": "wmar"},
           "data": {"classes": [1, 2], "per_class": 4, "eval_classes": [1, 2], "eval_per_class": 2},
           "imprint": {"phases": [{"steps": 3, "lr": 1e-3, "trainable": "full", "policy_lr": 0.1},
                                  {"steps": 2, "lr": 1e-3, "trainable": "full", "fix_rule": True}], "bs": 2},
           "detect": {"null": "base"}, "edits": ["image_roundtrip"], "quality": "none", "out": str(tmp_path / "run")}
    run_cfg(cfg, tmp_path)
    rep = check_report(tmp_path / "run")
    assert "image_roundtrip" in rep["edits"]


def geopt_cfg(out):
    return {"model": {"name": "geopt", "factory": "fakes:FakeGeoPT"},
            "scheme": {"name": "ripple", "params": {"eps": 0.05, "n_wrong": 20}},
            "data": {},
            "imprint": {"phases": [{"steps": 5, "lr": 1e-3, "trainable": "full"}]},
            "detect": {"null": "wrong_keys"}, "edits": ["finetune:steps=2,lr=1e-4"], "quality": "rel_l2",
            "out": str(out)}


def test_run_end_to_end_geopt(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path)
    check_report(tmp_path / "run")


def test_run_resumes_and_skips_done_steps(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx import pipeline
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path)
    os.remove(tmp_path / "run" / "edits.json")
    ran = []
    real = dict(pipeline.STEP_FNS)
    monkeypatch.setattr(pipeline, "STEP_FNS", {k: (lambda k: lambda ctx: (ran.append(k), real[k](ctx))[1])(k) for k in real})
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path)
    assert ran == ["edits", "quality", "report", "viewer"]


def test_partial_step_output_is_not_treated_as_done(tmp_path, monkeypatch):
    import markweightx.keys as K
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    out = tmp_path / "run"; out.mkdir()
    (out / "imprint.json.tmp").write_text("{}")            # a step killed while writing its output
    run_cfg(geopt_cfg(out), tmp_path, only=["prepare", "selfgen", "calibrate", "imprint"])
    assert json.load(open(out / "imprint.json"))["release"]


def test_config_rejects_unknown_keys_and_steps(tmp_path):
    from markweightx import pipeline
    with pytest.raises(pipeline.ConfigError, match="allowed: .*model"):
        pipeline.load_config(write_cfg(tmp_path, {**geopt_cfg(tmp_path / "r"), "modle": {}}))
    with pytest.raises(pipeline.ConfigError, match="prepare"):
        pipeline.run(pipeline.load_config(write_cfg(tmp_path, geopt_cfg(tmp_path / "r"))), only=["imprnt"])
    with pytest.raises(pipeline.ConfigError, match="edit"):
        pipeline.load_config(write_cfg(tmp_path, {**geopt_cfg(tmp_path / "r"), "edits": ["no_such_edit"]}))


def env(tmp_path):
    return dict(os.environ, PYTHONPATH=f"{ROOT}:{TESTS}", CUDA_VISIBLE_DEVICES="", MWX_KEYS=str(tmp_path / "keys"))


def test_cli_run_smoke(tmp_path):
    cfg = write_cfg(tmp_path, geopt_cfg(tmp_path / "run"))
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "run", cfg, "--smoke"], capture_output=True,
                       text=True, env=env(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert os.path.isfile(tmp_path / "run_smoke" / "report.json")      # a smoke run writes beside the real run


def test_detect_npz_with_geopt(tmp_path):
    cfg = write_cfg(tmp_path, geopt_cfg(tmp_path / "run"))
    e = env(tmp_path)
    assert subprocess.run([sys.executable, "-m", "markweightx.cli", "run", cfg, "--smoke"], env=e,
                          capture_output=True).returncode == 0
    from fakes import FakeGeoPT
    _, test = FakeGeoPT().splits()
    p, c, _ = test[0]
    from markweightx import pipeline                       # the marked model's output on this case
    rel = os.path.join(tmp_path, "run_smoke", "release")
    from markweightx.types import Release
    m = Release.open(rel).adapter_model(trust=True)          # the stand-in factory is code the release names
    field = m.run(m.load(Release.open(rel), "cpu"), [(p, c)])[0]
    np_path = tmp_path / "case.npz"
    import numpy as np
    np.savez(np_path, points=p.numpy(), field=field.detach().numpy(), condition=c.numpy())
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "detect", str(np_path), "--key", rel, "--model",
                        "geopt", "--release", rel, "--trust-plugin"], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(rows) == 1 and "p_value" in rows[0]


def test_detect_png_dir_with_taming(tmp_path):
    cfg = {"model": {"name": "taming", "factory": "fakes:FakeTaming"}, "scheme": {"name": "wmar"},
           "data": {"classes": [1], "per_class": 2, "eval_classes": [1], "eval_per_class": 2},
           "imprint": {"phases": [{"steps": 2, "lr": 1e-3, "trainable": "full"}], "bs": 2},
           "detect": {"null": "base"}, "edits": [], "quality": "none", "out": str(tmp_path / "run")}
    e = env(tmp_path)
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "run", write_cfg(tmp_path, cfg)], env=e,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    pngs = tmp_path / "run" / "assets" / "student"
    assert len(list(pngs.glob("*.png"))) == 2
    rel = os.path.join(tmp_path, "run", "release")
    r = subprocess.run([sys.executable, "-m", "markweightx.cli", "detect", str(pngs), "--key", rel, "--model",
                        "taming", "--release", rel, "--trust-plugin"], capture_output=True, text=True, env=e)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    assert len(rows) == 2 and all("p_value" in x for x in rows)


def test_config_reads_an_unquoted_null_key(tmp_path):
    from markweightx import pipeline
    p = tmp_path / "c.yaml"
    p.write_text("model: {name: geopt, factory: 'fakes:FakeGeoPT'}\nscheme: {name: ripple}\nout: x\n"
                 "detect: {fpr: 0.01, null: wrong_keys}\n")
    assert pipeline.load_config(str(p)).detect["null"] == "wrong_keys"


def test_smoke_caps_taming_classes(tmp_path):
    from markweightx import pipeline
    cfg = pipeline.load_config(write_cfg(tmp_path, {
        "model": {"name": "taming", "factory": "fakes:FakeTaming"}, "scheme": {"name": "wmar"}, "out": str(tmp_path / "r"),
        "data": {"classes": "0-999", "per_class": 20, "eval_classes": [1, 9, 232], "eval_per_class": 100}, "smoke": True}))
    prep = pipeline.step_prepare(pipeline.Context(cfg, "cpu"))
    assert len(prep["train_jobs"]) <= 4 and len(prep["eval_jobs"]) <= 4


def test_config_model_from_a_plugin_file(tmp_path):
    from markweightx import pipeline
    f = tmp_path / "my_model.py"
    f.write_text("import markweightx as mwx\n\n@mwx.register_model\nclass Mine(mwx.Model):\n"
                 "    name, unit, output = 'mine_test', 'point', mwx.Output.FIELD\n")
    cfg = pipeline.load_config(write_cfg(tmp_path, {"model": {"name": "mine_test", "source": str(f), "scale": 2},
                                                    "scheme": {"name": "ripple"}, "out": str(tmp_path / "r")}))
    ctx = pipeline.Context(cfg, "cpu")
    assert ctx.adapter.name == "mine_test" and ctx.adapter.params == {"scale": 2}


def test_smoke_writes_beside_the_real_run(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx import pipeline
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    cfg = pipeline.load_config(write_cfg(tmp_path, geopt_cfg(tmp_path / "run")))
    cfg.smoke = True
    pipeline.run(cfg, device="cpu", only=["prepare"])
    assert (tmp_path / "run_smoke" / "prepare.json").exists() and not (tmp_path / "run").exists()


def test_out_with_other_settings_is_refused(tmp_path, monkeypatch):
    import markweightx.keys as K
    from markweightx import pipeline
    monkeypatch.setattr(K, "STORE", str(tmp_path / "keys"))
    run_cfg(geopt_cfg(tmp_path / "run"), tmp_path, only=["prepare"])
    other = {**geopt_cfg(tmp_path / "run"), "scheme": {"name": "ripple", "params": {"eps": 0.08, "n_wrong": 20}}}
    with pytest.raises(pipeline.ConfigError, match="other settings"):
        run_cfg(other, tmp_path, only=["prepare"])
