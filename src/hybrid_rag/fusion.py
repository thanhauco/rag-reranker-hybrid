"""Rank / score fusion for hybrid retrieval.

Two families:

* **Rank-based** (RRF): ignores raw scores; robust because BM25 scores are unbounded
  while cosine scores live in [-1, 1]. The safe default.
* **Score-based** (convex combination, CombSUM/CombMNZ, DBSF): uses score magnitudes,
  so each list must be normalised first. Can beat RRF when weights are tuned on
  labelled data - and lose to it when they aren't.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .types import Document, SearchResult

ResultLists = Mapping[str, Sequence[SearchResult]]


# --------------------------------------------------------------------------- normalisers
def minmax(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def zscore(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2:
        return np.ones_like(scores)
    sd = scores.std()
    return (scores - scores.mean()) / sd if sd > 1e-12 else np.zeros_like(scores)


def dbsf(scores: np.ndarray) -> np.ndarray:
    """Distribution-Based Score Fusion normaliser (Qdrant): clip to mean +/- 3 sigma,
    then min-max into [0, 1]. Less sensitive to a single outlier than plain min-max."""
    if len(scores) < 2:
        return np.ones_like(scores)
    mu, sd = scores.mean(), scores.std()
    lo, hi = mu - 3 * sd, mu + 3 * sd
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


NORMALIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "minmax": minmax, "zscore": zscore, "dbsf": dbsf, "none": lambda s: s,
}


# --------------------------------------------------------------------------- helpers
def _collect(lists: ResultLists) -> tuple[dict[str, Document], dict[str, SearchResult]]:
    docs: dict[str, Document] = {}
    merged: dict[str, SearchResult] = {}
    for results in lists.values():
        for r in results:
            docs.setdefault(r.id, r.document)
            if r.id not in merged:
                merged[r.id] = SearchResult(r.document, 0.0, {}, {})
            merged[r.id].scores.update(r.scores)
            merged[r.id].ranks.update(r.ranks)
    return docs, merged


def _finalize(merged: dict[str, SearchResult], fused: dict[str, float], stage: str,
              k: int | None) -> list[SearchResult]:
    out = []
    for doc_id, score in fused.items():
        r = merged[doc_id]
        r.score = float(score)
        r.scores[stage] = float(score)
        out.append(r)
    out.sort(key=lambda r: (-r.score, r.id))
    for rank, r in enumerate(out, start=1):
        r.ranks[stage] = rank
    return out[:k] if k else out


def _weights(lists: ResultLists, weights: Mapping[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {name: 1.0 for name in lists}
    missing = set(lists) - set(weights)
    if missing:
        raise ValueError(f"weights missing for {sorted(missing)}")
    return dict(weights)


# --------------------------------------------------------------------------- fusion
def reciprocal_rank_fusion(lists: ResultLists, k: int = 60,
                           weights: Mapping[str, float] | None = None,
                           top_k: int | None = None) -> list[SearchResult]:
    """RRF (Cormack et al., SIGIR 2009): ``score(d) = sum_i w_i / (k + rank_i(d))``.

    ``k=60`` damps the advantage of the very top ranks; smaller ``k`` makes fusion
    more "winner takes all".
    """
    w = _weights(lists, weights)
    _, merged = _collect(lists)
    fused: dict[str, float] = {}
    for name, results in lists.items():
        for rank, r in enumerate(results, start=1):
            fused[r.id] = fused.get(r.id, 0.0) + w[name] / (k + rank)
    return _finalize(merged, fused, "rrf", top_k)


def _normalised(lists: ResultLists, normalization: str) -> dict[str, dict[str, float]]:
    norm = NORMALIZERS[normalization]
    out = {}
    for name, results in lists.items():
        s = norm(np.array([r.score for r in results], dtype=np.float64))
        out[name] = {r.id: float(v) for r, v in zip(results, s)}
    return out


def weighted_fusion(lists: ResultLists, weights: Mapping[str, float] | None = None,
                    normalization: str = "minmax", top_k: int | None = None,
                    stage: str = "weighted") -> list[SearchResult]:
    """Convex combination of normalised scores: ``sum_i w_i * norm(s_i(d))``.

    A document missing from list ``i`` gets that list's *worst* normalised score - the
    "not found" prior. (Using 0 would be wrong for z-scores, where 0 is the mean: a miss
    would outrank every below-average hit.)
    """
    w = _weights(lists, weights)
    _, merged = _collect(lists)
    norm = _normalised(lists, normalization)
    floors = {name: min(scores.values()) if scores else 0.0 for name, scores in norm.items()}
    fused = {
        doc_id: sum(w[name] * norm[name].get(doc_id, floors[name]) for name in norm)
        for doc_id in merged
    }
    return _finalize(merged, fused, stage, top_k)


def distribution_based_fusion(lists: ResultLists, weights: Mapping[str, float] | None = None,
                              top_k: int | None = None) -> list[SearchResult]:
    return weighted_fusion(lists, weights, "dbsf", top_k, stage="dbsf")


def comb_sum(lists: ResultLists, normalization: str = "minmax",
             top_k: int | None = None) -> list[SearchResult]:
    return weighted_fusion(lists, None, normalization, top_k, stage="combsum")


def comb_mnz(lists: ResultLists, normalization: str = "minmax",
             top_k: int | None = None) -> list[SearchResult]:
    """CombMNZ (Fox & Shaw, 1994): CombSUM x number of lists that retrieved the doc.
    Rewards agreement between retrievers."""
    _, merged = _collect(lists)
    norm = _normalised(lists, normalization)
    total: dict[str, float] = {}
    hits: dict[str, int] = {}
    for scores in norm.values():
        for doc_id, s in scores.items():
            total[doc_id] = total.get(doc_id, 0.0) + s
            hits[doc_id] = hits.get(doc_id, 0) + 1
    fused = {d: total[d] * hits[d] for d in total}
    return _finalize(merged, fused, "combmnz", top_k)


FUSION_METHODS = {
    "rrf": reciprocal_rank_fusion,
    "weighted": weighted_fusion,
    "dbsf": distribution_based_fusion,
    "combsum": comb_sum,
    "combmnz": comb_mnz,
}


def fuse(lists: ResultLists, method: str = "rrf", **kwargs) -> list[SearchResult]:
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion {method!r}; choose from {sorted(FUSION_METHODS)}")
    return FUSION_METHODS[method](lists, **kwargs)
