from transformers import LogitsProcessor
import torch
import numpy as np

from transformers import LogitsProcessor

class PROLogitsProcessor(LogitsProcessor):
    def __init__(self, watermark):
        self.watermark = watermark  

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = self.watermark.watermark_logits(input_ids, scores)
        return scores
