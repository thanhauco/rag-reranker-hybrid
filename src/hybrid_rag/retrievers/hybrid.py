"""Hybrid retrieval: run several retrievers in parallel and fuse their rankings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..fusion import fuse
from ..types import Document, SearchResult
from .base import Retriever

# Signals that a query wants exact lexical matching: error codes, versions, IDs, quotes.
_EXACT_SIGNALS = re.compile(
    r'"[^"]+"|\b[A-Z]{2,}[_\-]?[A-Z0-9_]+\b|\b\w+[_.]\w+\b|\b\d{3,}\b|\bv?\d+\.\d+(\.\d+)?\b'
)


def query_aware_alpha(query: str, base_alpha: float = 0.5, shift: float = 0.25) -> float:
    """Heuristic per-query lexical weight in [0, 1].

    Identifier-like tokens push weight toward BM25; long natural-language questions push
    it toward dense retrieval. Replace with a learned classifier once you have labels.
    """
    alpha = base_alpha
    if _EXACT_SIGNALS.search(query):
        alpha += shift
    n_words = len(query.split())
    if n_words >= 10:
        alpha -= shift / 2
    elif n_words <= 3:
        alpha += shift / 2
    return min(max(alpha, 0.0), 1.0)


class HybridRetriever(Retriever):
    """Fuse any number of first-stage retrievers.

    Each sub-retriever fetches ``candidate_k`` results (deeper than the final ``k``: fusion
    needs overlap to work) and the fused list is truncated to ``k``.

    ``fusion`` is one of ``rrf | weighted | dbsf | combsum | combmnz``. With
    ``dynamic_alpha=True`` and exactly two retrievers (lexical first), weights are set
    per query by :func:`query_aware_alpha`.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        fusion: str = "rrf",
        weights: Mapping[str, float] | None = None,
        candidate_k: int = 100,
        fusion_kwargs: Mapping[str, Any] | None = None,
        dynamic_alpha: bool = False,
        parallel: bool = True,
    ):
        super().__init__()
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = dict(retrievers)
        self.fusion = fusion
        self.weights = dict(weights) if weights else None
        self.candidate_k = candidate_k
        self.fusion_kwargs = dict(fusion_kwargs or {})
        self.dynamic_alpha = dynamic_alpha
        self.parallel = parallel
        if dynamic_alpha and len(self.retrievers) != 2:
            raise ValueError("dynamic_alpha requires exactly two retrievers (lexical, dense)")

    def index(self, documents: Iterable[Document]) -> HybridRetriever:
        self.documents = list(documents)
        if self.parallel and len(self.retrievers) > 1:
            with ThreadPoolExecutor(len(self.retrievers)) as ex:
                list(ex.map(lambda r: r.index(self.documents), self.retrievers.values()))
        else:
            for r in self.retrievers.values():
                r.index(self.documents)
        return self

    def retrieve_all(self, query: str, k: int | None = None) -> dict[str, list[SearchResult]]:
        k = k or self.candidate_k
        items = list(self.retrievers.items())
        if self.parallel and len(items) > 1:
            with ThreadPoolExecutor(len(items)) as ex:
                lists = list(ex.map(lambda kv: kv[1].search(query, k), items))
        else:
            lists = [r.search(query, k) for _, r in items]
        # namespace scores by the configured key so two retrievers of the same class
        # (e.g. two dense models) don't collide
        out: dict[str, list[SearchResult]] = {}
        for (key, r), results in zip(items, lists):
            if key != r.name:
                for res in results:
                    res.scores[key] = res.scores.pop(r.name, res.score)
                    res.ranks[key] = res.ranks.pop(r.name, 0)
            out[key] = results
        return out

    def _weights_for(self, query: str) -> dict[str, float] | None:
        if not self.dynamic_alpha:
            return self.weights
        lexical, dense = list(self.retrievers)
        alpha = query_aware_alpha(query, base_alpha=(self.weights or {}).get(lexical, 0.5))
        return {lexical: alpha, dense: 1 - alpha}

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        lists = self.retrieve_all(query)
        kwargs = dict(self.fusion_kwargs)
        weights = self._weights_for(query)
        if weights is not None and self.fusion in ("rrf", "weighted", "dbsf"):
            kwargs["weights"] = weights
        return fuse(lists, self.fusion, top_k=k, **kwargs)
