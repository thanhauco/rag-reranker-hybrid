"""Dense (bi-encoder) retrieval with an exact numpy index or FAISS HNSW."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..embeddings import Embedder, HashingEmbedder
from ..types import Document, SearchResult
from .base import Retriever


class VectorIndex:
    """Exact inner-product search. With L2-normalised vectors this is cosine similarity.

    Brute force is the right default up to ~1M vectors on CPU: no recall loss, no tuning.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def scores(self, query: np.ndarray) -> np.ndarray:
        return self.vectors @ query.astype(np.float32)

    def __len__(self) -> int:
        return len(self.vectors)


class FaissIndex(VectorIndex):
    """FAISS HNSW (approximate). Returns -inf for vectors outside the probed top ``k``."""

    def __init__(self, dim: int, m: int = 32, ef_search: int = 128):
        try:
            import faiss
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'hybrid-rag[faiss]'") from e
        super().__init__(dim)
        self._faiss = faiss
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.k = 256

    def add(self, vectors: np.ndarray) -> None:
        super().add(vectors)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def scores(self, query: np.ndarray) -> np.ndarray:
        out = np.full(len(self), -np.inf, dtype=np.float32)
        k = min(self.k, len(self))
        d, i = self.index.search(query.reshape(1, -1).astype(np.float32), k)
        valid = i[0] >= 0
        out[i[0][valid]] = d[0][valid]
        return out


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, embedder: Embedder | None = None, index: str = "flat",
                 include_title: bool = True):
        super().__init__()
        self.embedder = embedder or HashingEmbedder()
        self.index_type = index
        self.include_title = include_title
        self._index: VectorIndex | None = None

    def _make_index(self) -> VectorIndex:
        if self.index_type == "faiss":
            return FaissIndex(self.embedder.dim)
        return VectorIndex(self.embedder.dim)

    def _doc_text(self, d: Document) -> str:
        return f"{d.title}\n{d.text}" if self.include_title and d.title else d.text

    def index(self, documents: Iterable[Document]) -> DenseRetriever:
        self.documents = list(documents)
        self._index = self._make_index()
        if self.documents:
            texts = [self._doc_text(d) for d in self.documents]
            self._index.add(self.embedder.embed_documents(texts))
        return self

    def add(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if self._index is None:
            self._index = self._make_index()
        self._index.add(self.embedder.embed_documents([self._doc_text(d) for d in docs]))
        self.documents.extend(docs)

    def get_scores(self, query: str) -> np.ndarray:
        if self._index is None or not len(self._index):
            return np.zeros(0, dtype=np.float32)
        return self._index.scores(self.embedder.embed_query(query))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        results = self._to_results(self.get_scores(query), k)
        return [r for r in results if np.isfinite(r.score)]

    def save_vectors(self, path: str | Path) -> None:
        assert self._index is not None
        np.save(path, self._index.vectors)
