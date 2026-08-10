from hybrid_rag.rerankers import (
    CascadeReranker,
    ColBERTReranker,
    EnsembleReranker,
    HeuristicReranker,
)
from tests.helpers import CANDIDATES, QUERY


def test_cascade_funnels():
    c = CascadeReranker([(HeuristicReranker(), 2), (ColBERTReranker(), 1)])
    out = c.rerank(QUERY, CANDIDATES)
    assert len(out) == 1 and {"heuristic", "colbert"} <= set(out[0].scores)


def test_ensemble_rrf():
    e = EnsembleReranker({"h": HeuristicReranker(), "c": ColBERTReranker()})
    out = e.rerank(QUERY, CANDIDATES, top_k=2)
    assert out[0].id == "d2" and "ensemble" in out[0].scores
