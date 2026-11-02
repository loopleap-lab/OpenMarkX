# Tutorial: watermarking your own model

A model enters OpenMarkX through a **model adapter**, and a watermark through a **scheme**. A run config points at
both. This tutorial writes the two files for a model that predicts a field on 3D points, runs them, and points at
the GeoPT and Taming adapters as worked examples.

| Your model | What you write |
|---|---|
| a Llama-style causal language model | nothing: `--base <hf-id or local dir>` (see [Another base model](user-guide.md#another-base-model)) |
| an image and text model in the Gemma 3 style | nothing: the `vlm` adapter with `base:` |
| a field model on 3D points | a model class for the `geopt` adapter ([below](#running-a-field-model-today)) |
| another kind of model | a model adapter, `my_model.py` |

- [The three files](#the-three-files)
- [Step 1: the model adapter](#step-1-the-model-adapter)
- [Step 2: the mark](#step-2-the-mark)
- [Step 3: the config](#step-3-the-config)
- [What runs today](#what-runs-today)
- [Running a field model today](#running-a-field-model-today)
- [Worked examples: GeoPT and Taming](#worked-examples-geopt-and-taming)

## The three files

OpenMarkX itself does not change:

```
my_project/
├── my_model.py    how to load and run the model, and what one output unit is
├── my_mark.py     how the secret key shifts each unit, and how to score an output
└── run.yaml       points at both
```

## Step 1: the model adapter

A model adapter subclasses `omx.Model` and registers itself by name:

```python
# my_model.py
import torch

import openmarkx as omx


@omx.register_model
class MyModel(omx.Model):
    name = "my_model"
    unit = "point"                       # what one output unit is, for reports
    output = omx.Output.FIELD            # FIELD: trained by MSE to the scheme's target_output

    def __init__(self, ckpt: str | None = None):
        super().__init__(ckpt=ckpt)      # saved in the release, so detect rebuilds the same adapter

    def load(self, path, device):
        """The base (path None) or a release, as a torch module."""
        net = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
        weights = getattr(path, "weights", None) or self.params["ckpt"]
        if weights:
            net.load_state_dict(torch.load(weights, map_location=device))
        return net.to(device).eval()

    def run(self, net, inputs, **kw):
        """A batch of inputs in, outputs out."""
        return [net(points) for points in inputs]

    def forward(self, net, batch):
        """Teacher-forced pass for training: (inputs, output)."""
        return batch, net(batch["points"])
```

| Member | Meaning | Default |
|---|---|---|
| `name` | the adapter's name in a config | required |
| `unit` | what one output unit is | required |
| `output` | `Output.TOKENS` (trained by KL divergence to `target_logits`) or `Output.FIELD` (trained by mean squared error to `target_output`) | `TOKENS` |
| `load(path, device)` | the base (`path` is None) or a release, as a torch module | required |
| `run(net, inputs, **kw)` | a batch of inputs in, outputs out | required |
| `forward(net, batch)` | the teacher-forced pass for training: `(prefix ids, logits)` for tokens, `(inputs, output)` for fields | `net(input_ids=...)` for tokens, `net(batch["inputs"])` for fields |
| `codec()` | outputs to the units the scheme scores, and back (a tokenizer, a VQ) | the identity |
| `trainable(net, mode)` | the weights the imprint trains: `"lora"` or `"full"` | all parameters for `full` |
| `lora_targets` | module names LoRA wraps | the Llama projections |

Parameters in the config's `model:` block reach `__init__` and are saved in the release (`release.json` gains
`model: {name, params}`, `format: hf | state_dict` and `schema: 2`), so `detect` and `verify` rebuild the same
adapter. Code your adapter imports from someone else's checkout goes through
`openmarkx.external.require_external(name, ENV_VAR, module)`, which stops with a message naming the variable when
it is missing.

## Step 2: the mark

A mark on continuous outputs is an `omx.FieldScheme`. This one adds a keyed wave over the points and scores the
correlation of a suspect output with that wave:

```python
# my_mark.py
import torch

import openmarkx as omx


@omx.register_scheme
class MyMark(omx.FieldScheme):
    name = "my-mark"

    def __init__(self, eps: float = 0.05):
        super().__init__(eps=eps)

    def setup(self, model, codec, device, key=None):
        """Key: a secret for a new release, or the one in an existing key."""
        self.model, self.codec, self.device = model, codec, device
        key = dict(key or {})
        key.setdefault("secret", omx.keys.new_secret())
        g = torch.Generator().manual_seed(int(key["secret"]))
        self.w, self.b = torch.randn(3, 8, generator=g) * 4, torch.rand(8, generator=g) * 6.28
        return key

    def _wave(self, points):
        xyz = points[..., :3].float()
        return torch.cos(xyz @ self.w.to(xyz.device) + self.b.to(xyz.device)).mean(-1, keepdim=True)

    def target_output(self, inputs, output):
        """Signal: the unmarked output plus the keyed wave."""
        return output + self.params["eps"] * self._wave(inputs["points"]).to(output.dtype)

    def score(self, output, reference, inputs):
        """Score: correlation of (output - reference) with the wave, and the number of points."""
        r = (output - reference).float().flatten(0, -2).mean(-1)
        g = self._wave(inputs["points"]).flatten()
        r, g = r - r.mean(), g - g.mean()
        return float((r * g).sum() / (r.norm() * g.norm() + 1e-12)), g.numel()
```

`score` compares a suspect output with the unmarked model's output on the same inputs, so detection needs the
publisher's unmarked model (access level R1). The secret sits in the key under `secret`, so the default
`wrong_key` can redraw it for the wrong-key null. For a model with token outputs, subclass `omx.Scheme` instead and
write `target_logits` and `score` ([Methods](methods.md#adding-a-method)).

## Step 3: the config

```yaml
# run.yaml
model: {name: my_model, source: my_model.py}
scheme: {name: my-mark, source: my_mark.py, params: {eps: 0.05}}
imprint:
  phases:
    - {steps: 500, lr: 1.0e-4, trainable: full}
detect: {fpr: 0.01, null: wrong_keys}
out: runs/my_model
```

```bash
openmarkx run run.yaml --smoke
```

[Running an experiment from a config](user-guide.md#running-an-experiment-from-a-config) lists every key.

## What runs today

`openmarkx run` checks this config, imports both files and builds the adapter. Its steps from `prepare` on are
written for the five built-in adapters, so a new adapter name stops at the first step:

```
openmarkx run: no prepare step for model my_model
```

Running a new adapter through every step is on the [roadmap](../ROADMAP.md).

| Setup | What runs |
|---|---|
| a new adapter (`model: {name: my_model, source: my_model.py}`) | the config check; the run stops at `prepare` |
| your model behind the `geopt` adapter, with `scheme: ripple` | every step, and `openmarkx detect --model geopt` |
| your model behind the `geopt` adapter, with `my_mark.py` | every step through `report.json`; the `viewer` step and `detect --model geopt` call `ripple`'s own methods and stop with an error |

## Running a field model today

The `geopt` adapter takes any model behind a small class, named in the config with `factory: module:Class`. The
class gives the model, one forward call per case, the data, and a normalizer:

```python
# my_field.py
import torch


class Identity:
    def decode(self, y):
        return y


class MyField:
    def __init__(self, ckpt, data, device):
        """ckpt and data come from the config's model block."""
        self.net = torch.nn.Sequential(torch.nn.Linear(3 + 2, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
        if ckpt:
            self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.y_normalizer = Identity()        # decodes outputs to physical units, for quality: rel_l2

    def model(self):
        """The torch module whose weights the imprint trains."""
        return self.net

    def __call__(self, net, points, cond):
        """One case: points [1, N, 3] and a condition [1, 1, 2] in, a field [1, N, 1] out."""
        return net(torch.cat([points, cond.expand(-1, points.shape[1], -1)], -1))

    def splits(self):
        """(train, test): lists of (points, condition, true field)."""
        g = torch.Generator().manual_seed(0)
        case = lambda: (torch.randn(1, 200, 3, generator=g), torch.randn(1, 1, 2, generator=g),
                        torch.randn(1, 200, 1, generator=g))
        return [case() for _ in range(8)], [case() for _ in range(4)]
```

```yaml
# run.yaml
model: {name: geopt, factory: "my_field:MyField", ckpt: null, data: null}
scheme: {name: my-mark, source: my_mark.py, params: {eps: 0.05}}      # or {name: ripple, params: {eps: 0.05}}
imprint:
  phases:
    - {steps: 400, lr: 1.0e-3, trainable: full}
detect: {fpr: 0.01, null: wrong_keys, n_null: 20}
edits: ["finetune:steps=20,lr=1e-4"]
quality: rel_l2
out: runs/my_field
```

```bash
PYTHONPATH=. openmarkx run run.yaml                   # with my_mark.py, add: --only prepare selfgen calibrate imprint generate detect edits quality report
```

`PYTHONPATH=.` lets the adapter import `my_field`. The release names that code, so opening it outside its own run
needs `--trust-plugin`. This example runs on CPU. `tests/fakes.py` (`FakeGeoPT`) is another class of this shape,
and `tests/test_pipeline.py` runs it through every step.

## Worked examples: GeoPT and Taming

| | GeoPT | Taming |
|---|---|---|
| adapter | [`openmarkx/models/geopt.py`](../openmarkx/models/geopt.py) | [`openmarkx/models/taming.py`](../openmarkx/models/taming.py) |
| scheme | [`openmarkx/schemes/ripple.py`](../openmarkx/schemes/ripple.py) | [`openmarkx/schemes/wmar.py`](../openmarkx/schemes/wmar.py) |
| demo config | [`configs/demos/geopt.yaml`](../configs/demos/geopt.yaml), [`geopt_colearn.yaml`](../configs/demos/geopt_colearn.yaml) | [`configs/demos/taming.yaml`](../configs/demos/taming.yaml) |
| model | a Transolver physics surrogate, 6 output channels on aircraft surface points for a flight condition | Taming Transformers' class-conditional image generator: a VQ and a GPT over 16x16 image tokens |
| output, unit | `FIELD`, surface point | `TOKENS`, image token |
| external code | the GeoPT repo, `OMX_GEOPT_DIR` | WMAR (with Taming), `OMX_WMAR_DIR` |
| imprint | full fine-tune; optionally the mark written into the last layer in closed form first (`edit_last`) | full fine-tune of the GPT, in two phases: the green lists learned with the model, then fixed |
| codec | none | the VQ: images to codes and back |
| detection | against the publisher's unmarked model on the same inputs, with a wrong-key null | images encoded with the release's VQ, against the key's frozen null |

Both adapters keep the authors' code unchanged in your own checkout and load it with `require_external`. Each wraps
it behind a small class (`_GeoPT`, WMAR's wrapper), which the `factory` parameter replaces with a stand-in; the
CPU tests use the stand-ins in `tests/fakes.py`.
