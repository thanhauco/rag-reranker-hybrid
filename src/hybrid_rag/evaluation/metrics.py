"""Standard IR metrics. ``ranked`` is a list of doc ids, best first; ``qrels`` maps doc id
-> graded relevance (> 0 means relevant)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, int]


def _relevant(qrels: Qrels) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    rel = _relevant(qrels)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    if k <= 0:
        return 0.0
    rel = _relevant(qrels)
    return sum(1 for d in ranked[:k] if d in rel) / k


def hit_rate_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    rel = _relevant(qrels)
    return float(any(d in rel for d in ranked[:k]))


def mrr_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    rel = _relevant(qrels)
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    # exponential gain (2^rel - 1) as in Burges et al. / TREC; equals linear for binary
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    gains = [max(qrels.get(d, 0), 0) for d in ranked[:k]]
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def average_precision(ranked: Sequence[str], qrels: Qrels, k: int | None = None) -> float:
    rel = _relevant(qrels)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k] if k else ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision,
}
