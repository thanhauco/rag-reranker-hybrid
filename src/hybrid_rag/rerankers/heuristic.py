"""A fast, model-free reranker built from classic IR features.

Signals that first-stage BM25 ignores: query-term *coverage* (BM25 can rank a passage
that repeats one term above one that matches all of them), term *proximity*, phrase
(bigram) matches, and title matches. Blended with the normalised first-stage score.
Runs in microseconds - a good first cascade stage or a no-GPU fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..text import tokenize
from ..types import Document, SearchResult
from .base import Reranker


def _min_window(positions: dict[str, list[int]]) -> int | None:
    """Smallest span (in tokens) containing at least one occurrence of every term."""
    if not positions:
        return None
    events = sorted((p, t) for t, ps in positions.items() for p in ps)
    need, have, best, left = len(positions), {}, None, 0
    for pos, term in events:
        have[term] = have.get(term, 0) + 1
        while len(have) == need:
            span = pos - events[left][0] + 1
            best = span if best is None else min(best, span)
            lt = events[left][1]
            have[lt] -= 1
            if have[lt] == 0:
                del have[lt]
            left += 1
    return best


class HeuristicReranker(Reranker):
    name = "heuristic"

    def __init__(self, w_coverage: float = 0.4, w_proximity: float = 0.2,
                 w_phrase: float = 0.15, w_title: float = 0.1, w_prior: float = 0.35,
                 top_k: int | None = None, min_score: float | None = None):
        super().__init__(top_k, min_score)
        self.w = dict(coverage=w_coverage, proximity=w_proximity, phrase=w_phrase,
                      title=w_title, prior=w_prior)

    def features(self, query: str, doc: Document) -> dict[str, float]:
        q_toks = tokenize(query)
        q_terms = list(dict.fromkeys(q_toks))
        if not q_terms:
            return dict(coverage=0.0, proximity=0.0, phrase=0.0, title=0.0)
        d_toks = tokenize(f"{doc.title} {doc.text}" if doc.title else doc.text)
        positions: dict[str, list[int]] = {}
        for i, t in enumerate(d_toks):
            if t in q_terms:
                positions.setdefault(t, []).append(i)
        # longer terms tend to be more specific: weight coverage by log length
        weights = {t: math.log(2 + len(t)) for t in q_terms}
        coverage = sum(weights[t] for t in positions) / sum(weights.values())
        window = _min_window(positions)
        proximity = 0.0
        if window is not None and len(positions) > 1:
            proximity = len(positions) / window
        elif window is not None:
            proximity = 1.0 / len(q_terms)
        q_bi = set(zip(q_toks, q_toks[1:]))
        phrase = (len(q_bi & set(zip(d_toks, d_toks[1:]))) / len(q_bi)) if q_bi else 0.0
        t_toks = set(tokenize(doc.title))
        title = len(t_toks & set(q_terms)) / len(q_terms) if t_toks else 0.0
        return dict(coverage=coverage, proximity=proximity, phrase=phrase, title=title)

    def score(self, query: str, documents: Sequence[Document],
              prior: Sequence[float] | None = None) -> list[float]:
        prior = prior if prior is not None else [0.0] * len(documents)
        out = []
        for d, p in zip(documents, prior):
            f = self.features(query, d)
            out.append(sum(self.w[k] * v for k, v in f.items()) + self.w["prior"] * p)
        return out

    def rerank(self, query: str, results: Sequence[SearchResult],
               top_k: int | None = None) -> list[SearchResult]:
        if not results:
            return []
        s = np.array([r.score for r in results], dtype=np.float64)
        span = s.max() - s.min()
        prior = ((s - s.min()) / span).tolist() if span > 1e-12 else [1.0] * len(results)
        scores = self.score(query, [r.document for r in results], prior)
        return self._finalize([r.with_stage(self.name, v) for r, v in zip(results, scores)], top_k)
