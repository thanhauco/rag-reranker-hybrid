"""Neural hybrid search: BGE bi-encoder + BM25, then cross-encoder / ColBERT reranking.

    pip install -e ".[models]"
    python examples/03_neural_rerankers.py
"""

import time

from hybrid_rag import BM25Retriever, DenseRetriever, HybridRetriever, SentenceTransformerEmbedder
from hybrid_rag.evaluation import evaluate, load_sample, to_markdown
from hybrid_rag.rerankers import (
    CascadeReranker,
    ColBERTReranker,
    CrossEncoderReranker,
    HeuristicReranker,
    HFTokenEncoder,
)

ds = load_sample()
embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
hybrid = HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever(embedder)})
hybrid.index(ds.corpus)

rerankers = {
    "cross-encoder minilm": CrossEncoderReranker("minilm"),
    "cross-encoder bge-base": CrossEncoderReranker("bge-base"),
    # late interaction over a real encoder's token embeddings
    "colbert (bge-small tokens)": ColBERTReranker(HFTokenEncoder("BAAI/bge-small-en-v1.5")),
    # cheap stage prunes 30 -> 10 before the expensive one
    "cascade heuristic->bge": CascadeReranker([(HeuristicReranker(), 10),
                                               (CrossEncoderReranker("bge-base"), None)]),
}

query = "why does my container get killed when it uses too much RAM"
cands = hybrid.search(query, 30)
print(f"Q: {query}\nfirst stage: {[c.id for c in cands[:5]]}")
for name, rr in rerankers.items():
    t = time.perf_counter()
    out = rr.rerank(query, cands, 5)
    print(f"  {name:<28} {[r.id for r in out]}  {(time.perf_counter() - t) * 1000:.0f} ms")

systems = {"hybrid": lambda q: hybrid.search(q, 20)}
for name, rr in rerankers.items():
    systems[f"+ {name}"] = (lambda r: lambda q: r.rerank(q, hybrid.search(q, 30), 20))(rr)
print("\n" + to_markdown([evaluate(fn, ds, system=n) for n, fn in systems.items()]))
