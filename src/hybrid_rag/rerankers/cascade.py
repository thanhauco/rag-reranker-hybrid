"""Composing rerankers.

* :class:`CascadeReranker` - multi-stage funnel. Each stage sees only the survivors of
  the previous one: e.g. 200 candidates -> heuristic -> 50 -> cross-encoder -> 10 ->
  LLM listwise -> 5. Cost is dominated by the last (smallest) stage.
* :class:`EnsembleReranker` - run several rerankers on the same candidates and fuse
  their rankings (RRF by default), which smooths out any single model's blind spots.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..fusion import fuse
from ..types import Document, SearchResult
from .base import Reranker


class CascadeReranker(Reranker):
    name = "cascade"

    def __init__(self, stages: Sequence[tuple[Reranker, int | None]]):
        super().__init__()
        if not stages:
            raise ValueError("CascadeReranker needs at least one stage")
        self.stages = list(stages)

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        raise NotImplementedError("CascadeReranker is set-wise; call rerank()")

    def rerank(self, query: str, results: Sequence[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        current = list(results)
        for reranker, keep in self.stages:
            current = reranker.rerank(query, current, keep)
        k = top_k or self.top_k
        return current[:k] if k else current


class EnsembleReranker(Reranker):
    name = "ensemble"

    def __init__(self, rerankers: Mapping[str, Reranker], method: str = "rrf",
                 weights: Mapping[str, float] | None = None, top_k: int | None = None):
        super().__init__(top_k)
        self.rerankers = dict(rerankers)
        self.method, self.weights = method, weights

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        raise NotImplementedError("EnsembleReranker is set-wise; call rerank()")

    def rerank(self, query: str, results: Sequence[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        lists = {name: r.rerank(query, results) for name, r in self.rerankers.items()}
        kwargs = {"weights": self.weights} if self.weights else {}
        fused = fuse(lists, self.method, **kwargs)
        for rank, r in enumerate(fused, start=1):
            r.scores[self.name] = r.score
            r.ranks[self.name] = rank
        k = top_k or self.top_k
        return fused[:k] if k else fused
