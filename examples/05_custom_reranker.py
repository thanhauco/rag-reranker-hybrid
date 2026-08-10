"""Plug in your own reranker: implement ``score()`` and it works everywhere (cascades,
ensembles, the pipeline, the evaluator, YAML-free).

    python examples/05_custom_reranker.py
"""

from collections.abc import Sequence
from datetime import date

from hybrid_rag import BM25Retriever, DenseRetriever, Document, HybridRetriever
from hybrid_rag.rerankers import CascadeReranker, EnsembleReranker, HeuristicReranker, Reranker


class FreshnessReranker(Reranker):
    """Blend relevance with recency - a common business rule for docs/support search."""

    name = "freshness"

    def __init__(self, half_life_days: float = 365, weight: float = 0.3, **kw):
        super().__init__(**kw)
        self.half_life, self.weight = half_life_days, weight

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        today = date(2026, 10, 1)
        out = []
        for d in documents:
            age = (today - date.fromisoformat(d.metadata.get("updated", "2020-01-01"))).days
            out.append(0.5 ** (age / self.half_life))
        return out


docs = [
    Document("old", "To rotate TLS certificates, restart the ingress controller manually.",
             {"title": "TLS rotation (legacy)", "updated": "2021-03-01"}),
    Document("new", "TLS certificates rotate automatically with cert-manager; no restart needed.",
             {"title": "TLS rotation", "updated": "2026-08-15"}),
    Document("other", "DNS TTL controls how long resolvers cache records.",
             {"title": "DNS", "updated": "2026-01-10"}),
]
hybrid = HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()}).index(docs)
query = "how do I rotate TLS certificates"
cands = hybrid.search(query, 10)

print("first stage :", [r.id for r in cands])
print("heuristic   :", [r.id for r in HeuristicReranker().rerank(query, cands)])
ens = EnsembleReranker({"relevance": HeuristicReranker(), "fresh": FreshnessReranker()},
                       weights={"relevance": 1.0, "fresh": 0.7})
print("ensemble    :", [r.id for r in ens.rerank(query, cands)])
cascade = CascadeReranker([(HeuristicReranker(), 2), (FreshnessReranker(), 1)])
print("cascade     :", [r.id for r in cascade.rerank(query, cands)])
