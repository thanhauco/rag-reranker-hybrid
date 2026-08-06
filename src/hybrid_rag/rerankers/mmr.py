"""Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

Greedy selection balancing relevance against redundancy:

    MMR = argmax_d [ lambda * rel(q, d) - (1 - lambda) * max_{s in S} sim(d, s) ]

Run it *after* a relevance reranker: it uses the incoming scores as ``rel`` so the
context window isn't filled with five near-duplicate chunks of the same page.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..embeddings import Embedder, HashingEmbedder
from ..types import Document, SearchResult
from .base import Reranker


class MMRReranker(Reranker):
    name = "mmr"

    def __init__(self, embedder: Embedder | None = None, lambda_: float = 0.7,
                 top_k: int | None = None, use_incoming_scores: bool = True):
        super().__init__(top_k)
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        self.embedder = embedder or HashingEmbedder()
        self.lambda_ = lambda_
        self.use_incoming_scores = use_incoming_scores

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        q = self.embedder.embed_query(query)
        return (self.embedder.embed_documents([d.text for d in documents]) @ q).tolist()

    def rerank(self, query: str, results: Sequence[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        if not results:
            return []
        k = top_k or self.top_k or len(results)
        vecs = self.embedder.embed_documents([r.text for r in results])
        if self.use_incoming_scores:
            rel = np.array([r.score for r in results], dtype=np.float64)
        else:
            rel = vecs @ self.embedder.embed_query(query)
        span = rel.max() - rel.min()
        rel = (rel - rel.min()) / span if span > 1e-12 else np.ones_like(rel)
        sim = vecs @ vecs.T

        selected: list[int] = []
        remaining = list(range(len(results)))
        out: list[SearchResult] = []
        while remaining and len(selected) < k:
            if selected:
                redundancy = sim[np.ix_(remaining, selected)].max(axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            mmr = self.lambda_ * rel[remaining] - (1 - self.lambda_) * redundancy
            best = int(np.argmax(mmr))
            idx = remaining.pop(best)
            selected.append(idx)
            out.append(results[idx].with_stage(self.name, float(mmr[best]), len(selected)))
        return out
