"""train.imprint: the generic imprint loop for token (KL) and field (MSE) schemes, phases, fix_rule and resume."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from fakes import tiny_field_net, tiny_llama
from markweightx.models.base import Model, Output
from markweightx.schemes.base import FieldScheme, Scheme


class Green(Scheme):
    """Adds delta to the first half of the vocabulary; with colearn, the bias itself is a trainable parameter."""
    name = "green_test"

    def __init__(self, delta=4.0, colearn=False):
        super().__init__(delta=delta, colearn=colearn)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=colearn)

    def target_logits(self, input_ids, logits):
        v = logits.shape[-1]
        add = torch.zeros(v); add[: v // 2] = self.params["delta"]
        return logits + add + self.bias

    def trainable_params(self):
        return [self.bias] if self.params["colearn"] else []

    def aux_loss(self):
        return (self.bias - 1.0).pow(2).sum() if self.params["colearn"] else None


class Shift(FieldScheme):
    name = "shift_test"

    def target_output(self, inputs, output):
        return output + 0.1

    def score(self, output, reference, inputs):
        return float((output - reference).mean()), output.numel()


class TinyText(Model):
    name, unit, output = "tiny_text_test", "token", Output.TOKENS


class TinyField(Model):
    name, unit, output = "tiny_field_test", "point", Output.FIELD


def token_batches(n=4):
    g = torch.Generator().manual_seed(0)
    return [{"input_ids": torch.randint(0, 1000, (2, 12), generator=g)} for _ in range(n)]


def kl_on(net, sch, batch):
    from markweightx.imprint import kl_teacher_student
    with torch.no_grad():
        ids = batch["input_ids"]
        lg = net(input_ids=ids).logits[:, :-1].float()
        if hasattr(net, "disable_adapter"):
            with net.disable_adapter():
                base = net(input_ids=ids).logits[:, :-1].float()
        else:
            base = lg
        return kl_teacher_student(sch.target_logits(ids[:, :-1], base), lg).item()


def test_kl_phase_moves_student_toward_teacher(tmp_path):
    from markweightx.train import imprint
    net, sch, b = tiny_llama(), Green(), token_batches()
    before = kl_on(net, sch, b[0])
    rel = imprint(TinyText(), sch, b, str(tmp_path), net=net, device="cpu",
                  phases=[{"steps": 20, "lr": 5e-3, "trainable": "lora", "rank": 8}])
    after = kl_on(rel.net, sch, b[0])
    assert after < before
    assert rel.model["name"] == "tiny_text_test" and rel.format == "hf"


def test_mse_phase_learns_field_target(tmp_path):
    from markweightx.train import imprint
    net, sch = tiny_field_net(), Shift()
    g = torch.Generator().manual_seed(0)
    b = [{"inputs": torch.randn(64, 3, generator=g)} for _ in range(4)]
    ref = [net(x["inputs"]).detach() for x in b]
    before = sum(torch.nn.functional.mse_loss(r, r + 0.1).item() for r in ref)
    rel = imprint(TinyField(), sch, b, str(tmp_path), net=net, device="cpu",
                  phases=[{"steps": 50, "lr": 3e-2, "trainable": "full"}])
    after = sum(torch.nn.functional.mse_loss(rel.net(x["inputs"]), r + 0.1).item() for x, r in zip(b, ref))
    assert after < 0.5 * before
    assert rel.format == "state_dict" and os.path.isfile(rel.weights)


def test_fix_rule_phase_freezes_scheme_params(tmp_path):
    from markweightx.train import imprint
    net, sch, b = tiny_llama(), Green(colearn=True), token_batches()
    seen = {}
    imprint(TinyText(), sch, b, str(tmp_path), net=net, device="cpu",
            phases=[{"steps": 5, "lr": 1e-3, "trainable": "lora", "rank": 4, "policy_lr": 0.1}],
            on_phase_end=lambda i: seen.setdefault(i, sch.bias.detach().clone()))
    assert not torch.equal(seen[0], torch.zeros(1))             # co-learning moved the rule
    imprint(TinyText(), sch, b, str(tmp_path / "b"), net=tiny_llama(), device="cpu",
            phases=[{"steps": 3, "lr": 1e-3, "trainable": "lora", "rank": 4, "policy_lr": 0.1},
                    {"steps": 3, "lr": 1e-3, "trainable": "lora", "rank": 4, "fix_rule": True}],
            on_phase_end=lambda i: seen.setdefault(("b", i), sch.bias.detach().clone()))
    assert torch.equal(seen[("b", 0)], seen[("b", 1)])          # fixed during phase 2


def test_resume_continues_from_saved_step(tmp_path):
    from markweightx.train import imprint
    p1 = {"steps": 3, "lr": 1e-3, "trainable": "lora", "rank": 4}
    p2 = {"steps": 2, "lr": 1e-3, "trainable": "lora", "rank": 4}
    imprint(TinyText(), Green(), token_batches(), str(tmp_path), net=tiny_llama(), device="cpu", phases=[p1])
    log = tmp_path / "train_log.jsonl"
    n1 = len(log.read_text().splitlines())
    imprint(TinyText(), Green(), token_batches(), str(tmp_path), net=tiny_llama(), device="cpu", phases=[p1, p2])
    new = [json.loads(l) for l in log.read_text().splitlines()[n1:]]
    assert n1 == 3 and [r["phase"] for r in new] == [1, 1]


def test_zero_step_phase_writes_the_release(tmp_path):
    """GeoPT s1_e02: the mark is written in closed form, then a 0-step phase (run_mark.py's max(steps, 1))."""
    from markweightx.train import imprint
    rel = imprint(TinyField(), Shift(), [{"inputs": torch.randn(8, 3)}], str(tmp_path), net=tiny_field_net(),
                  device="cpu", phases=[{"steps": 0, "lr": 1e-5, "trainable": "full"}])
    assert os.path.isfile(rel.weights)
