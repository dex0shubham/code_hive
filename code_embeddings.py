"""
Code-aware embedding similarity using transformer models like UniXcoder /
CodeBERT.

These models are trained on code (and code+nl pairs) and tend to embed
semantically similar snippets close together even when surface text differs,
which is a stronger "logic-level" signal than generic NL sentence embeddings.

Default model: microsoft/unixcoder-base (~500MB on first download).

Usage:
    from code_embeddings import CodeEmbeddingComputer
    enc = CodeEmbeddingComputer()
    sim = enc.cosine_sim_matrix(list_of_code_strings)
"""

from __future__ import annotations

import numpy as np


class CodeEmbeddingComputer:
    """Encode code snippets with a code-aware transformer; mean-pooled embeddings."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "code_embeddings requires `transformers` and `torch`. "
                "Install with `pip install transformers torch`."
            ) from e

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        torch = self._torch
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.model(**tokens)
            last_hidden = outputs.last_hidden_state  # (B, T, D)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = (summed / counts).cpu().numpy()
            all_embs.append(mean)
        return np.vstack(all_embs)

    def cosine_sim_matrix(self, codes: list[str]) -> np.ndarray:
        embs = self.encode(codes)
        if embs.shape[0] == 0:
            return np.zeros((0, 0))
        norm = np.linalg.norm(embs, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-9, None)
        normed = embs / norm
        return normed @ normed.T

    def mean_pairwise_similarity(self, codes: list[str]) -> float:
        if len(codes) < 2:
            return 0.0
        sim = self.cosine_sim_matrix(codes)
        iu = np.triu_indices_from(sim, k=1)
        return float(sim[iu].mean()) if iu[0].size else 0.0
