from .watermark_processor import WatermarkContext
PRO_TYPE = "pro"   # was watermarks.watermark_types.WatermarkType.PRO
import torch
from typing import List

class PROWatermark:
    def __init__(
        self,
        device,
        tokenizer,
        delta=4.0,
        gamma=0.5,
        chunk_size=10,
        embedding_model="perceptiveshawty/compositional-bert-large-uncased",
        transform_model_path=None,
        mapping_file_path=None
    ):
        self.type = PRO_TYPE
        self.model = WatermarkContext(
            device=device,
            chunk_length=chunk_size,
            tokenizer=tokenizer,
            delta=delta,
            gamma=gamma,
            embedding_model=embedding_model,
            transform_model_path=transform_model_path,
            mapping_file_path=mapping_file_path
        )
        self.transform_model = self.model.transform_model
        self.last_transformed_embeddings = None
        self.last_input_ids = None
        self.device = device
        self._mapping_tensor = None

    def set_tanh_temperature(self, tau: float):
        """Set the tanh temperature for scale_vector. Lower = smoother gradients."""
        self.model.tanh_temperature = tau

    def get_tanh_temperature(self) -> float:
        return self.model.tanh_temperature

    def _get_mapping_tensor(self, device):
        if self._mapping_tensor is None or self._mapping_tensor.device != device:
            self._mapping_tensor = torch.tensor(self.model.mapping, device=device, dtype=torch.long)
        return self._mapping_tensor

    def get_embeddings(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Compute BERT embeddings and transform them. Caches result for reuse."""
        if (
            self.last_transformed_embeddings is not None
            and self.last_input_ids is not None
            and input_ids.shape == self.last_input_ids.shape
            and torch.equal(input_ids, self.last_input_ids)
        ):
            return self.last_transformed_embeddings

        self.last_input_ids = input_ids
        embeddings = self.model.get_batch_embeddings(input_ids)
        transformed = self.transform_model(embeddings)
        self.last_transformed_embeddings = transformed
        return transformed

    def watermark_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        anti: bool = False
    ) -> torch.FloatTensor:
        mapping_tensor = self._get_mapping_tensor(logits.device)

        # Case 1: generation (logits: [batch, vocab_size])
        if logits.dim() == 2:
            transformed = self.get_embeddings(input_ids)
            scaled = self.model.scale_vector(transformed)
            bias = scaled[:, mapping_tensor]
            sign = -1.0 if anti else 1.0
            return logits + sign * bias * self.model.delta

        # Case 2: training (logits: [batch, seq_len, vocab_size])
        transformed = self.get_embeddings(input_ids)
        scaled_batch = self.model.scale_vector(transformed)
        bias_batch = scaled_batch[:, mapping_tensor]
        bias_batch = bias_batch.unsqueeze(1)

        sign = -1.0 if anti else 1.0
        return logits + sign * bias_batch * self.model.delta

    def detect(self, text: str):
        return self.model.detect(text)

    def detect_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        return self.model.detect_batch(texts, batch_size=batch_size)
