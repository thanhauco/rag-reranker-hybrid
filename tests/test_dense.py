import numpy as np
import pytest

from hybrid_rag import BM25Retriever, DenseRetriever, HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a, b = e.embed_query("liveness probe"), e.embed_query("liveness probe")
    assert np.allclose(a, b) and np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_dense_is_typo_tolerant(docs):
    r = DenseRetriever(HashingEmbedder()).index(docs)
    assert r.search("kubrnetes livness probb", k=1)[0].id == "c"
    assert BM25Retriever().index(docs).search("kubrnetes livness probb") == []


def test_dense_incremental_add(docs):
    r = DenseRetriever().index(docs[:2])
    r.add(docs[2:])
    assert len(r) == len(docs)
    assert r.search("redis maxmemory", k=1)[0].id == "e"
