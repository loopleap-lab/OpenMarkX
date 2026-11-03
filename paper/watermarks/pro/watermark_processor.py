import os
import json
import random
import numpy as np
import torch
import jieba
import torch.nn as nn
from transformers import LogitsProcessor, AutoTokenizer, AutoModel
from .transform_model import TransformModel
from typing import List
from tqdm import tqdm

class PROWatermarkBase:
    def __init__(self, gamma: float, delta: float, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.gamma = gamma
        self.delta = delta

class WatermarkContext(PROWatermarkBase):
    def __init__(
        self,
        device: torch.device,
        chunk_length: int,
        tokenizer,
        delta: float = 4.0,
        gamma: float = 0.5,
        embedding_model: str = "perceptiveshawty/compositional-bert-large-uncased",
        transform_model_path: str = None,
        mapping_file_path: str = None
    ):
        super().__init__(gamma, delta, tokenizer)
        self.device = device
        self.chunk_length = chunk_length

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.embedding_model.eval()

        self.transform_model = TransformModel()
        if transform_model_path and os.path.exists(transform_model_path):
            state_dict = torch.load(transform_model_path, map_location='cpu', weights_only=True)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            self.transform_model.load_state_dict(new_state_dict)
        else:
            self.transform_model.apply(self._init_weights)
        
        self.transform_model = self.transform_model.to(self.device)
        self.transform_model.train()

        # Temperature for tanh scaling. Low = smooth (gradients flow), high = sharp (strong watermark).
        # During co-optimization, anneal from tanh_temperature_init to tanh_temperature_max.
        # For frozen inference/detection, use 1000 (hard ±1).
        self.tanh_temperature = 1000.0

        model_for_access = self.transform_model
        watermark_dim = model_for_access.layers[-1].out_features

        if mapping_file_path is None:
            mapping_file_path = os.path.join(os.path.dirname(__file__), f"300_mapping_{len(tokenizer)}.json")
        mapping_file = mapping_file_path
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)
        else:
            self.mapping = [random.randint(0, watermark_dim - 1) for _ in range(len(tokenizer))]
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            with open(mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=4)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_embedding(self, sentence):
        enc = self.embedding_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**enc)
        return outputs.last_hidden_state[:, 0, :]

    def get_batch_embeddings(self, input_ids: torch.LongTensor):
        sentences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        context_sentences = []
        for sentence in sentences:
            words = list(jieba.cut(sentence))
            non_space_indices = [i for i, word in enumerate(words) if word.strip()]
            words_2d, chunk_start = [], 0
            for i in range(0, len(non_space_indices), self.chunk_length):
                chunk_end = min(i + self.chunk_length, len(non_space_indices))
                chunk_indices = non_space_indices[:chunk_end]
                if chunk_indices:
                    chunk = words[chunk_start:chunk_indices[-1] + 1]
                    words_2d.append(chunk)
                    chunk_start = chunk_indices[-1] + 1
            
            if len(words_2d) > 0:
                if len(words_2d[-1]) == self.chunk_length:
                    ctx = ''.join([''.join(group) for group in words_2d])
                else:
                    ctx = ''.join([''.join(group) for group in words_2d[:-1]])
            else:
                ctx = ""
            context_sentences.append(ctx)

        enc = self.embedding_tokenizer(
            context_sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**enc)
        return outputs.last_hidden_state[:, 0, :]

    def get_text_split(self, sentence):
        words = list(jieba.cut(sentence))
        non_space_indices = [i for i, word in enumerate(words) if word.strip()]
        words_2d, chunk_start = [], 0
        for i in range(0, len(non_space_indices), self.chunk_length):
            chunk_end = min(i + self.chunk_length, len(non_space_indices))
            chunk_indices = non_space_indices[:chunk_end]
            if chunk_indices:
                chunk = words[chunk_start:chunk_indices[-1] + 1]
                words_2d.append(chunk)
                chunk_start = chunk_indices[-1] + 1
        return words_2d

    def get_context_sentence(self, input_ids: torch.LongTensor):
        sentence = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        words_2d = self.get_text_split(sentence)
        if len(words_2d) > 0 and len(words_2d[-1]) == self.chunk_length:
            return ''.join([''.join(group) for group in words_2d])
        elif len(words_2d) > 0:
            return ''.join([''.join(group) for group in words_2d[:-1]])
        return ""

    def scale_vector(self, v, temperature=None):
        """Scale vector with tanh. Uses self.tanh_temperature if temperature not specified."""
        tau = temperature if temperature is not None else self.tanh_temperature
        if isinstance(v, torch.Tensor):
            v = v - v.mean(dim=-1, keepdim=True)
            return torch.tanh(tau * v)
        elif isinstance(v, np.ndarray):
            v = v - np.mean(v)
            return np.tanh(tau * v)
        else:
            raise TypeError("Unsupported input type for scale_vector")

    def _get_bias(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get bias for decoding-time watermarking. Always uses hard temperature."""
        context_sentence = self.get_context_sentence(input_ids)
        context_embedding = self.get_embedding(context_sentence)
        output = self.transform_model(context_embedding)[0]
        scaled = self.scale_vector(output, temperature=1000.0)
        mapped = scaled[self.mapping]
        return mapped

    def detect_batch(self, texts: List[str], batch_size: int = 512) -> List[float]:
        all_pairs = []
        text_indices = []
        for tidx, text in enumerate(texts):
            words_2d = self.get_text_split(text)
            for i in range(1, len(words_2d)):
                context = ''.join(sum(words_2d[:i], [])).strip()
                if len(list(jieba.cut(context))) < 40:
                    continue
                current = ''.join(words_2d[i]).strip()
                all_pairs.append((context, current))
                text_indices.append(tidx)

        if not all_pairs:
            return [0.0] * len(texts)

        contexts, currents = zip(*all_pairs)
        B = len(contexts)

        mapping_tensor = torch.tensor(self.mapping, dtype=torch.long, device=self.device)
        all_mapped = []

        for start in tqdm(range(0, B, batch_size)):
            end = min(start + batch_size, B)
            sub_ctx = list(contexts[start:end])

            enc = self.embedding_tokenizer(
                sub_ctx,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.embedding_model(**enc)
                out = outputs.last_hidden_state[:, 0, :]

            model = self.transform_model.module if hasattr(self.transform_model, "module") else self.transform_model
            raw = model(out)
            scaled = self.scale_vector(raw, temperature=1000.0)

            mapped = scaled[:, mapping_tensor]
            all_mapped.append(mapped.cpu())

        all_mapped = torch.cat(all_mapped, dim=0)

        per_text_vals = [[] for _ in texts]
        pad_id = self.tokenizer.pad_token_id or 0

        for start in tqdm(range(0, B, batch_size)):
            end = min(start + batch_size, B)
            sub_curr = list(currents[start:end])

            tok = self.tokenizer(
                sub_curr,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            )
            token_ids = tok["input_ids"]

            for j in range(token_ids.size(0)):
                idx = start + j
                ids = token_ids[j]
                mask = ids.ne(pad_id)
                ids = ids[mask]
                vals = (all_mapped[idx, ids]).tolist()
                per_text_vals[text_indices[idx]].extend(vals)

        z_scores = [float(np.mean(v)) if v else 0.0 for v in per_text_vals]
        return z_scores

    def detect(self, text: str) -> float:
        word_2d = self.get_text_split(text)
        values = []
        for i in range(1, len(word_2d)):
            context_sentence = ''.join([''.join(group) for group in word_2d[:i]]).strip()
            current_sentence = ''.join(word_2d[i]).strip()
            if len(list(jieba.cut(context_sentence))) < 40:
                continue
            context_embedding = self.get_embedding(context_sentence)
            output = self.transform_model(context_embedding).cpu()[0].detach().numpy()
            similarity_array = self.scale_vector(output, temperature=1000.0)[self.mapping]
            tokens = self.tokenizer.encode(current_sentence, return_tensors="pt", add_special_tokens=False)
            for idx in range(len(tokens[0])):
                values.append(float(similarity_array[tokens[0][idx]]))
        return np.mean(values) if values else 0.0

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, watermark_model: WatermarkContext):
        self.watermark_model = watermark_model

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bias_batch = []
        for b in range(input_ids.shape[0]):
            bias = self.watermark_model._get_bias(input_ids[b])
            bias_batch.append(bias)
        bias_tensor = torch.stack(bias_batch).to(scores.device)
        return scores + bias_tensor * self.watermark_model.delta

