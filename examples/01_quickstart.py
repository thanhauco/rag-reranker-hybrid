"""Quickstart: hybrid retrieval + reranking in ~20 lines, fully offline.

    python examples/01_quickstart.py
"""

from hybrid_rag import BM25Retriever, DenseRetriever, Document, HybridRetriever, RAGPipeline
from hybrid_rag.rerankers import HeuristicReranker

docs = [
    Document("k8s-oom", "When a container exceeds its memory limit the OOM killer terminates "
             "it and the pod shows OOMKilled with exit code 137.", {"title": "OOMKilled"}),
    Document("k8s-probe", "A failing liveness probe restarts the container. A readiness probe "
             "only removes the pod from Service endpoints.", {"title": "Probes"}),
    Document("pg-conn", "FATAL: sorry, too many clients already means max_connections was "
             "reached. Put PgBouncer in front of Postgres.", {"title": "Postgres connections"}),
    Document("redis", "allkeys-lru evicts the least recently used keys when Redis reaches "
             "maxmemory.", {"title": "Redis eviction"}),
]

retriever = HybridRetriever(
    {"bm25": BM25Retriever(), "dense": DenseRetriever()},  # dense defaults to offline hashing
    fusion="rrf",
)
pipeline = RAGPipeline(retriever, reranker=HeuristicReranker(), retrieve_k=10, rerank_k=2)
pipeline.index(docs)

for query in ["pod restarted with exit code 137", "too many clients already", "kubrnetes probe"]:
    resp = pipeline.search(query)
    print(f"\nQ: {query}   ({resp.timings_ms['total']:.2f} ms)")
    for r in resp.results:
        trail = "  ".join(f"{stage}#{rank}" for stage, rank in r.ranks.items())
        print(f"  {r.score:6.3f}  {r.id:<10} {trail}")
