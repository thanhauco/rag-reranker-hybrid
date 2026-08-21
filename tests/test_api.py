import pytest

from hybrid_rag import BM25Retriever, ExtractiveGenerator, RAGPipeline, load_documents
from hybrid_rag.mock import generate_mock_dataset, write_markdown
from hybrid_rag.rerankers import HeuristicReranker


def test_api_endpoints(sample):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from hybrid_rag.api import create_app

    pipe = RAGPipeline(BM25Retriever(), HeuristicReranker(), ExtractiveGenerator()).index(
        sample.corpus)
    client = TestClient(create_app(pipe))
    assert client.get("/health").json()["documents"] == len(sample.corpus)
    r = client.post("/search", json={"query": "redis maxmemory eviction", "rerank_k": 2}).json()
    assert r["results"][0]["id"] == "d09" and len(r["results"]) == 2
    assert client.post("/answer", json={"query": "allkeys-lru"}).json()["answer"]["text"]
    assert client.post("/search", json={"query": ""}).status_code == 422
    n = client.post("/index", json={"documents": [{"id": "x", "text": "hello world"}]}).json()
    assert n == {"indexed": 1}



def test_api_index_endpoint_accepts_ingested_documents(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from hybrid_rag.api import create_app

    ds = generate_mock_dataset(n_services=3, n_problems=2, n_queries=6)
    write_markdown(ds, tmp_path)
    pipe = RAGPipeline(BM25Retriever(), HeuristicReranker())
    client = TestClient(create_app(pipe))
    docs = load_documents(tmp_path)
    r = client.post("/index", json={"documents": [
        {"id": d.id, "text": d.text, "metadata": d.metadata} for d in docs]})
    assert r.json() == {"indexed": 18}
    target = ds.corpus[5]
    res = client.post("/search", json={"query": target.metadata["error_code"],
                                       "rerank_k": 3}).json()
    assert res["results"][0]["metadata"]["error_code"] == target.metadata["error_code"]
