import pytest

from hybrid_rag.types import Document


@pytest.fixture(scope="session")
def sample():
    from hybrid_rag.evaluation import load_sample

    return load_sample()


@pytest.fixture
def docs():
    return [
        Document("a", "The quick brown fox jumps over the lazy dog.", {"title": "Fox"}),
        Document("b", "PostgreSQL error: FATAL sorry too many clients already.",
                 {"title": "Postgres"}),
        Document("c", "Kubernetes restarts containers that fail their liveness probe.",
                 {"title": "Probes"}),
        Document("d", "A fox is a small omnivorous mammal of the dog family.",
                 {"title": "Foxes"}),
        Document("e", "Redis evicts keys with allkeys-lru when maxmemory is reached."),
    ]


@pytest.fixture
def hybrid(docs):
    from hybrid_rag import BM25Retriever, DenseRetriever, HybridRetriever

    return HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()}).index(docs)
