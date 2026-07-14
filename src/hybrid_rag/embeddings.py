"""Embedding models behind one small protocol.

``HashingEmbedder`` is deterministic and offline (feature-hashed char n-grams). It is not
semantic - use it for tests, CI and as a typo-tolerant "fuzzy lexical" signal. For real
dense retrieval use ``SentenceTransformerEmbedder`` (e.g. BGE, E5, GTE, nomic).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .text import char_ngrams, tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _stable_hash(s: str) -> int:
    # Python's hash() is salted per process; use a stable digest instead.
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    """Signed feature hashing of word unigrams + char n-grams, L2-normalised."""

    def __init__(self, dim: int = 1024, n_min: int = 3, n_max: int = 5, word_weight: float = 2.0):
        self.dim, self.n_min, self.n_max, self.word_weight = dim, n_min, n_max, word_weight

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = [(g, 1.0) for g in char_ngrams(text, self.n_min, self.n_max)]
        feats += [(f"w:{t}", self.word_weight) for t in tokenize(text)]
        for f, w in feats:
            h = _stable_hash(f)
            v[h % self.dim] += w if (h >> 63) & 1 else -w
        # sublinear damping so long texts don't dominate
        v = np.sign(v) * np.log1p(np.abs(v))
        return l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


# Instruction prefixes recommended by each model family's authors. Asymmetric models
# perform noticeably worse without them.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
    "nomic": ("search_query: ", "search_document: "),
    "gte": ("", ""),
}


class SentenceTransformerEmbedder:
    """Wraps any ``sentence-transformers`` bi-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install 'hybrid-rag[models]' to use SentenceTransformerEmbedder"
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if query_prefix is None or doc_prefix is None:
            family = next((f for f in KNOWN_PREFIXES if f in model_name.lower()), None)
            qp, dp = KNOWN_PREFIXES.get(family, ("", "")) if family else ("", "")
            query_prefix = qp if query_prefix is None else query_prefix
            doc_prefix = dp if doc_prefix is None else doc_prefix
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.batch_size, self.normalize = batch_size, normalize
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension  # sentence-transformers < 6
        self.dim = int(get_dim())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class CachedEmbedder:
    """Memoises document embeddings by content hash (re-indexing becomes ~free)."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.dim = inner.dim
        self._cache: dict[str, np.ndarray] = {}

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vecs = self.inner.embed_documents([texts[i] for i in missing])
            for i, v in zip(missing, vecs):
                self._cache[keys[i]] = v
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[k] for k in keys])

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


def get_embedder(name: str = "hashing", **kwargs) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(**kwargs)
    if name in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder {name!r}")
