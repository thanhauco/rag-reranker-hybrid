from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..types import Document, SearchResult


class Reranker(ABC):
    """Second-stage scorer: expensive, high-precision re-ordering of a short candidate list.

    Pointwise rerankers only implement :meth:`score`; listwise / set-wise rerankers
    (LLM listwise, MMR) override :meth:`rerank` directly.
    """

    name: str = "reranker"

    def __init__(self, top_k: int | None = None, min_score: float | None = None):
        self.top_k = top_k
        self.min_score = min_score

    @abstractmethod
    def score(self, query: str, documents: Sequence[Document]) -> list[float]: ...

    def rerank(self, query: str, results: Sequence[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        if not results:
            return []
        scores = self.score(query, [r.document for r in results])
        if len(scores) != len(results):
            raise RuntimeError(f"{self.name}: got {len(scores)} scores for {len(results)} docs")
        out = [r.with_stage(self.name, s) for r, s in zip(results, scores)]
        return self._finalize(out, top_k)

    def _finalize(self, results: list[SearchResult], top_k: int | None) -> list[SearchResult]:
        # stable sort: ties keep first-stage order
        results = sorted(results, key=lambda r: -r.score)
        if self.min_score is not None:
            results = [r for r in results if r.score >= self.min_score]
        for rank, r in enumerate(results, start=1):
            r.ranks[self.name] = rank
        k = top_k if top_k is not None else self.top_k
        return results[:k] if k else results


class IdentityReranker(Reranker):
    """No-op baseline: keeps first-stage order. Useful for ablations."""

    name = "identity"

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        n = len(documents)
        return [float(n - i) for i in range(n)]
