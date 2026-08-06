from hybrid_rag.rerankers import MMRReranker
from tests.helpers import CANDIDATES, make_results


def test_mmr_removes_near_duplicates():
    cands = make_results(
        "Redis eviction allkeys-lru removes least recently used keys.",
        "Redis eviction allkeys-lru removes least recently used keys!",
        "Redis persistence uses RDB snapshots and AOF logs.",
    )
    out = MMRReranker(lambda_=0.5).rerank("redis", cands, top_k=2)
    assert [r.id for r in out] == ["d0", "d2"]


def test_mmr_lambda_one_is_pure_relevance():
    out = MMRReranker(lambda_=1.0).rerank("q", CANDIDATES)
    assert [r.id for r in out] == ["d0", "d1", "d2"]
