import pytest

from hybrid_rag import BM25Retriever, DenseRetriever, HashingEmbedder, HybridRetriever
from hybrid_rag.retrievers import query_aware_alpha


def test_hybrid_merges_and_records_stage_scores(hybrid):
    res = hybrid.search("fox dog", k=3)
    top = res[0]
    assert {"bm25", "dense", "rrf"} <= set(top.scores)
    assert top.ranks["rrf"] == 1


def test_hybrid_recovers_what_bm25_misses(hybrid):
    # typo query: BM25 finds nothing, hybrid still answers via the dense leg
    assert hybrid.search("kubrnetes livness probb", k=1)[0].id == "c"


def test_hybrid_namespaces_duplicate_retriever_types(docs):
    h = HybridRetriever({"dense_a": DenseRetriever(HashingEmbedder(dim=128)),
                         "dense_b": DenseRetriever(HashingEmbedder(dim=512))}).index(docs)
    top = h.search("fox", k=1)[0]
    assert {"dense_a", "dense_b"} <= set(top.scores)


@pytest.mark.parametrize("fusion", ["rrf", "weighted", "dbsf", "combsum", "combmnz"])
def test_hybrid_all_fusions(docs, fusion):
    h = HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()},
                        fusion=fusion).index(docs)
    assert h.search("too many clients postgres", k=1)[0].id == "b"


def test_query_aware_alpha():
    assert query_aware_alpha("ERR_CONN_RESET") > 0.5
    long_q = "why do my pods keep restarting every time the database has a brief outage"
    assert query_aware_alpha(long_q) < 0.5


def test_dynamic_alpha_requires_two(docs):
    with pytest.raises(ValueError):
        HybridRetriever({"bm25": BM25Retriever()}, dynamic_alpha=True)
