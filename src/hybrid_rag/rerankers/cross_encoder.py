"""Cross-encoder reranking: the query and passage are encoded *jointly*, so every query
token attends to every passage token. Far more accurate than a bi-encoder, but cost is
O(candidates) forward passes per query - hence reranking only the top 20-100."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import Document
from .base import Reranker

PRESETS = {
    # ~22M params, English, very fast; the classic MS MARCO baseline
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # BAAI rerankers: strong quality; v2-m3 is multilingual with 8k context
    "bge-base": "BAAI/bge-reranker-base",
    "bge-large": "BAAI/bge-reranker-large",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "mxbai-base": "mixedbread-ai/mxbai-rerank-base-v1",
    "jina-v2": "jinaai/jina-reranker-v2-base-multilingual",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "minilm",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        sigmoid: bool = True,
        include_title: bool = True,
        trust_remote_code: bool = False,
        top_k: int | None = None,
        min_score: float | None = None,
    ):
        super().__init__(top_k, min_score)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'hybrid-rag[models]' to use CrossEncoderReranker") from e
        self.model_name = PRESETS.get(model_name, model_name)
        self.model = CrossEncoder(self.model_name, max_length=max_length, device=device,
                                  trust_remote_code=trust_remote_code)
        self.batch_size = batch_size
        self.sigmoid = sigmoid
        self.include_title = include_title

    def _passage(self, d: Document) -> str:
        return f"{d.title}. {d.text}" if self.include_title and d.title else d.text

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        if not documents:
            return []
        import torch

        pairs = [(query, self._passage(d)) for d in documents]
        kwargs = {"batch_size": self.batch_size, "show_progress_bar": False}
        try:  # sentence-transformers >= 4: request raw logits explicitly
            out = self.model.predict(pairs, activation_fn=torch.nn.Identity(), **kwargs)
        except TypeError:  # pragma: no cover - older versions
            out = self.model.predict(pairs, **kwargs)
        raw = np.asarray(out, dtype=np.float64).reshape(len(pairs), -1)[:, -1]
        # Sigmoid maps logits to [0, 1] so a ``min_score`` threshold means roughly the same
        # thing across queries (ranking is unchanged - sigmoid is monotonic).
        return (_sigmoid(raw) if self.sigmoid else raw).tolist()
