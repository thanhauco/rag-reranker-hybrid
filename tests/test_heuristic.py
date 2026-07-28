
from hybrid_rag.rerankers import HeuristicReranker, IdentityReranker
from hybrid_rag.rerankers.heuristic import _min_window
from tests.helpers import CANDIDATES, QUERY


def test_identity_keeps_order():
    out = IdentityReranker().rerank("q", CANDIDATES)
    assert [r.id for r in out] == ["d0", "d1", "d2"]


def test_heuristic_beats_keyword_stuffing():
    out = HeuristicReranker().rerank(QUERY, CANDIDATES)
    assert out[0].id == "d2"
    assert out[0].ranks["heuristic"] == 1 and "heuristic" in out[0].scores


def test_min_window():
    assert _min_window({"a": [0, 10], "b": [3, 11]}) == 2
    assert _min_window({"a": [5]}) == 1
    assert _min_window({}) is None


def test_reranker_top_k_and_min_score():
    out = HeuristicReranker(min_score=0.5).rerank(QUERY, CANDIDATES, top_k=1)
    assert len(out) <= 1 and all(r.score >= 0.5 for r in out)
