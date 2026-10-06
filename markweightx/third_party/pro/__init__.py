"""The PRO watermark (NeurIPS 2026) as released with the paper: the semantic watermark policy, its logits processor
and its initial policy weights (transform_model_cbert.pth). Copied from the PRO paper code (built on Gu et al.'s
watermark-learnability, Apache-2.0, see LICENSE in this directory) at commit 72b92f2; the only change is that
pro_watermark.py no longer imports the shared WatermarkType enum."""
from .logits_processor import PROLogitsProcessor
from .pro_watermark import PROWatermark

__all__ = ["PROWatermark", "PROLogitsProcessor"]
