"""Compare every fusion method on one query and show *why* documents moved.

    python examples/02_fusion_comparison.py "my postgres is running out of connections"
"""

import sys

from hybrid_rag import BM25Retriever, DenseRetriever, fuse
from hybrid_rag.evaluation import load_sample

query = sys.argv[1] if len(sys.argv) > 1 else "my postgres is running out of connections"
ds = load_sample()
bm25 = BM25Retriever().index(ds.corpus)
dense = DenseRetriever().index(ds.corpus)
lists = {"bm25": bm25.search(query, 20), "dense": dense.search(query, 20)}

print(f"query: {query!r}\n")
for name, results in lists.items():
    print(f"{name:>8}: " + "  ".join(f"{r.id}({r.score:.2f})" for r in results[:5]))

configs = {
    "rrf k=60": ("rrf", {"k": 60}),
    "rrf k=10": ("rrf", {"k": 10}),
    "weighted 0.5/0.5": ("weighted", {"weights": {"bm25": 0.5, "dense": 0.5}}),
    "weighted 0.2/0.8": ("weighted", {"weights": {"bm25": 0.2, "dense": 0.8}}),
    "weighted zscore": ("weighted", {"normalization": "zscore"}),
    "dbsf": ("dbsf", {}),
    "combsum": ("combsum", {}),
    "combmnz": ("combmnz", {}),
}
print()
for label, (method, kw) in configs.items():
    fused = fuse(lists, method, top_k=5, **kw)
    cells = []
    for r in fused:
        b, d = r.ranks.get("bm25", "-"), r.ranks.get("dense", "-")
        cells.append(f"{r.id}[b{b}/d{d}]")
    print(f"{label:>17}: " + "  ".join(cells))

relevant = ds.qrels.get("q02", {})
print(f"\n(for the default query, relevant docs are {sorted(relevant)})")
