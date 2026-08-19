"""Generate a mock knowledge base as Markdown files, ingest it like real data, and see
where each retrieval strategy wins - broken down by query kind.

    python examples/06_ingest_mock_data.py
"""

import tempfile
from pathlib import Path

from hybrid_rag import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RAGPipeline,
    SentenceChunker,
    generate_mock_dataset,
    ingest_path,
)
from hybrid_rag.evaluation import evaluate, group_metrics
from hybrid_rag.mock import write_markdown
from hybrid_rag.rerankers import HeuristicReranker

ds = generate_mock_dataset(n_services=12, n_queries=80, seed=0)
print(ds, "| query kinds:", sorted(set(ds.query_types.values())))

with tempfile.TemporaryDirectory() as tmp:
    docs_dir = write_markdown(ds, Path(tmp) / "docs")      # 360 .md files with YAML front matter
    systems = {
        "bm25": RAGPipeline(BM25Retriever(), retrieve_k=20, rerank_k=20),
        "dense": RAGPipeline(DenseRetriever(), retrieve_k=20, rerank_k=20),
        "hybrid rrf": RAGPipeline(
            HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()}),
            retrieve_k=50, rerank_k=20),
        "hybrid + heuristic": RAGPipeline(
            HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()}),
            HeuristicReranker(), chunker=SentenceChunker(max_words=120),
            retrieve_k=50, rerank_k=20),
    }
    reports = {}
    for name, pipe in systems.items():
        stats = ingest_path(pipe, docs_dir)                   # <- the real ingestion path
        reports[name] = evaluate(lambda q, p=pipe: p.search(q).results, ds,
                                 ["ndcg@10"], system=name)
        if name == "bm25":
            print("ingested:", stats.summary())

kinds = sorted(set(ds.query_types.values()))
print(f"\n{'nDCG@10':<20}" + "".join(f"{k:>12}" for k in kinds) + f"{'all':>10}")
for name, rep in reports.items():
    g = group_metrics(rep, ds.query_types)
    print(f"{name:<20}" + "".join(f"{g[k]['ndcg@10']:>12.3f}" for k in kinds)
          + f"{rep.metrics['ndcg@10']:>10.3f}")
