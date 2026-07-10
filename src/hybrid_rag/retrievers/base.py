from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np

from ..types import Document, SearchResult


class Retriever(ABC):
    """First-stage retriever: cheap, high-recall candidate generation."""

    name: str = "retriever"

    def __init__(self) -> None:
        self.documents: list[Document] = []

    @abstractmethod
    def index(self, documents: Iterable[Document]) -> Retriever: ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]: ...

    def search_batch(self, queries: Sequence[str], k: int = 10) -> list[list[SearchResult]]:
        return [self.search(q, k) for q in queries]

    def __len__(self) -> int:
        return len(self.documents)

    def _to_results(self, scores: np.ndarray, k: int) -> list[SearchResult]:
        """Top-k by score using argpartition (O(n) instead of a full sort)."""
        n = len(scores)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        out = []
        for rank, i in enumerate(idx, start=1):
            s = float(scores[i])
            out.append(SearchResult(self.documents[i], s, {self.name: s}, {self.name: rank}))
        return out
