"""A scheme says how a watermark is learned into the weights (the training signal) and how a text is scored."""
from __future__ import annotations

import torch

from ..primitives import Access, Imprint


class Scheme:
    """A watermarking method, composed from the six primitives (see markweightx.primitives):
    Key (setup), Signal (target_logits), Imprint (`imprint`), Perturb (edits; `booster` for training-time steps),
    Probe (`access`), Score (score / score_batch). Subclass it in one file and decorate with @register_scheme."""
    name: str = "base"
    kind: str = "distill_logit"   # distill_logit | distill_sampling | none
    imprint: Imprint = Imprint.DISTILL
    access: Access = Access.R0    # what the detector needs besides the text
    teacher: str | None = None    # upstream teacher name, when the scheme distills one
    experimental = False          # hyperparameter variants and unfinished schemes: hidden from describe()
    probe = None                  # a primitives.Probe, set by the SDK before detection (R1/R2 schemes use it)
    rng_bound = True              # randomness drawn on the model's device: detect on the device type the key was made on
    exact_null = False            # True: pvalue() is exact, so it is used even when the key carries an empirical null

    @property
    def needs_model_to_detect(self) -> bool:
        return self.access in (Access.R1, Access.R2)

    def __init__(self, **params):
        self.params = params

    # ---- lifecycle -------------------------------------------------------------------------
    def setup(self, model, tokenizer, device: str, key: dict | None = None) -> dict:
        """Bind to a model/tokenizer (teacher and detector state). Returns the key to store with the release."""
        self.model, self.tok, self.device = model, tokenizer, device
        return key or {}

    # ---- training signal ---------------------------------------------------------------------
    def target_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Teacher-forced teacher logits [B, T, V] for predicting token t+1 from the prefix ending at t."""
        raise NotImplementedError

    def trainable_params(self) -> list:
        """Extra parameters trained with the student (e.g. PRO's transform)."""
        return []

    def param_groups(self) -> list[dict]:
        """Optimizer groups for the extra parameters; default: same settings as the LoRA weights."""
        ps = self.trainable_params()
        return [{"params": ps}] if ps else []

    def aux_loss(self):
        return None

    def on_step(self, step: int, steps: int):
        pass

    def wrong_key(self, params: dict, rng, workdir: str) -> dict | None:
        """A wrong key for the wrong-key null: the true key's material with only its secret redrawn from `rng` (a
        random.Random), so wrong keys and the true key are exchangeable. Files it needs go in `workdir` (private,
        removed afterwards). None: this scheme has no secret to redraw."""
        if "secret" not in params:
            return None
        return {**params, "secret": rng.getrandbits(40)}

    def save(self, key_dir: str, key: dict) -> dict:
        """Persist key material the detector needs (e.g. a trained transform) as files in key_dir, a private
        directory in the key store (never the release), and return the key with their paths."""
        return key

    # ---- detection ---------------------------------------------------------------------------
    def score(self, text: str) -> tuple[float, int]:
        """(statistic, n_tokens); higher statistic = more evidence of the watermark."""
        raise NotImplementedError

    def score_batch(self, texts) -> list[tuple[float, int]]:
        return [self.score(t) for t in texts]

    def pvalue(self, statistic: float, n_tokens: int) -> float | None:
        """A closed-form p-value under the null, if the scheme has one; None = use the key's empirical null."""
        return None

    # ---- Imprint.EDIT (teacher-free schemes) ----------------------------------------------------
    def imprint_edit(self, model, tokenizer, out: str) -> str | None:
        """Write the mark into `model`'s weights directly and save the edited model under `out/weights`; return
        that directory (None = the release is the base itself). Key material (e.g. the perturbation) must be
        written by save() into its private key_dir, so it lands in the key store, never in the release."""
        raise NotImplementedError(f"{self.name}: imprint=EDIT schemes must implement imprint_edit()")


class NoWatermark(Scheme):
    """The unwatermarked base: a control row in every comparison."""
    name, kind = "nowm", "none"
    imprint = Imprint.EDIT
    rng_bound = False

    def imprint_edit(self, model, tokenizer, out):
        return None

    def score(self, text):
        return 0.0, len(self.tok(text, add_special_tokens=False)["input_ids"])
