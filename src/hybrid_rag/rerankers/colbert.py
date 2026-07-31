"""Late-interaction (ColBERT-style) reranking.

Query and passage are encoded *independently* into one vector per token; relevance is
MaxSim: for every query token, take its best-matching passage token, then sum.

    score(q, d) = sum_i max_j  q_i . d_j

Passage token matrices can be precomputed and cached, so reranking costs one query
encode + a few matrix products - a middle ground between bi- and cross-encoders.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..embeddings import HashingEmbedder, l2_normalize
from ..text import tokenize
from ..types import Document
from .base import Reranker


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    if len(query_tokens) == 0 or len(doc_tokens) == 0:
        return 0.0
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())


class TokenEncoder(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...

    def encode_document(self, text: str) -> np.ndarray: ...


class HashingTokenEncoder:
    """Offline token encoder: each word -> hashed char-n-gram vector.

    MaxSim over these behaves like soft, typo-tolerant term matching. Use it for tests or
    as a cheap baseline; use :class:`HFTokenEncoder` for real late interaction.
    """

    def __init__(self, dim: int = 256):
        self._emb = HashingEmbedder(dim=dim, word_weight=0.0)
        self._cache: dict[str, np.ndarray] = {}

    def _tok(self, t: str) -> np.ndarray:
        if t not in self._cache:
            self._cache[t] = self._emb.embed_query(t)
        return self._cache[t]

    def _encode(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        if not toks:
            return np.zeros((0, self._emb.dim), dtype=np.float32)
        return np.stack([self._tok(t) for t in toks])

    encode_query = _encode
    encode_document = _encode


class HFTokenEncoder:
    """Per-token embeddings from any Hugging Face encoder (last hidden state, L2-normalised,
    special/padding tokens dropped).

    Works with ColBERT checkpoints' backbones; for production ColBERT (with its trained
    projection head, query augmentation and PLAID indexing) use PyLate or RAGatouille and
    plug them in through the :class:`TokenEncoder` protocol.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str | None = None,
                 max_length: int = 256, query_max_length: int = 32):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'hybrid-rag[models]' to use HFTokenEncoder") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length, self.query_max_length = max_length, query_max_length

    def _encode(self, text: str, max_length: int) -> np.ndarray:
        enc = self.tokenizer(text, truncation=True, max_length=max_length,
                             return_tensors="pt", return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask")[0].bool()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            hidden = self.model(**enc).last_hidden_state[0]
        keep = (~special).to(hidden.device)
        vecs = hidden[keep].float().cpu().numpy()
        return l2_normalize(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text, self.query_max_length)

    def encode_document(self, text: str) -> np.ndarray:
        return self._encode(text, self.max_length)


class ColBERTReranker(Reranker):
    name = "colbert"

    def __init__(self, encoder: TokenEncoder | None = None, normalize: bool = True,
                 cache_documents: bool = True, top_k: int | None = None,
                 min_score: float | None = None):
        super().__init__(top_k, min_score)
        self.encoder = encoder or HashingTokenEncoder()
        self.normalize = normalize
        self.cache_documents = cache_documents
        self._doc_cache: dict[str, np.ndarray] = {}

    def _doc_tokens(self, d: Document) -> np.ndarray:
        if self.cache_documents and d.id in self._doc_cache:
            return self._doc_cache[d.id]
        toks = self.encoder.encode_document(f"{d.title} {d.text}".strip())
        if self.cache_documents:
            self._doc_cache[d.id] = toks
        return toks

    def precompute(self, documents: Sequence[Document]) -> None:
        """Encode passages offline (at index time) so query-time cost is minimal."""
        for d in documents:
            self._doc_tokens(d)

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        q = self.encoder.encode_query(query)
        denom = max(len(q), 1) if self.normalize else 1
        return [maxsim(q, self._doc_tokens(d)) / denom for d in documents]
