import math

import pytest

from hybrid_rag.evaluation import (
    average_precision,
    evaluate,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    to_markdown,
)

RANKED = ["x", "a", "y", "b"]
QRELS = {"a": 2, "b": 1, "c": 1}


def test_recall_precision_hit():
    assert recall_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, QRELS, 4) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, QRELS, 4) == pytest.approx(0.5)
    assert hit_rate_at_k(RANKED, QRELS, 1) == 0.0
    assert hit_rate_at_k(RANKED, QRELS, 2) == 1.0


def test_mrr():
    assert mrr_at_k(RANKED, QRELS, 10) == pytest.approx(0.5)
    assert mrr_at_k(RANKED, QRELS, 1) == 0.0


def test_ndcg_graded():
    dcg = (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(5)
    idcg = (2**2 - 1) / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, QRELS, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(1.0)


def test_average_precision():
    assert average_precision(RANKED, QRELS) == pytest.approx((1 / 2 + 2 / 4) / 3)


def test_evaluate_on_sample(sample, hybrid):
    from hybrid_rag import BM25Retriever

    bm25 = BM25Retriever().index(sample.corpus)
    rep = evaluate(lambda q: bm25.search(q, 10), sample, ["ndcg@10", "recall@10"], "bm25")
    assert 0.5 < rep.metrics["ndcg@10"] <= 1.0
    assert set(rep.per_query) == set(sample.queries)
    assert "| bm25 |" in to_markdown([rep])
    assert 0.0 <= paired_bootstrap(rep, rep, "ndcg@10", n=200) <= 1.0


def test_evaluate_maps_chunks_to_parents(sample):
    from hybrid_rag import BM25Retriever, FixedSizeChunker

    chunks = FixedSizeChunker(chunk_size=20, overlap=5).split(sample.corpus)
    assert len(chunks) > len(sample.corpus)
    bm25 = BM25Retriever().index(chunks)
    rep = evaluate(lambda q: bm25.search(q, 30), sample, ["recall@10"])
    assert rep.metrics["recall@10"] > 0.5
