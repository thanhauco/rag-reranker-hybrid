"""Ablation benchmark: lexical vs dense vs hybrid fusions vs rerankers.

    python benchmarks/run_benchmark.py                                 # offline, sample set
    python benchmarks/run_benchmark.py --dataset mock --models         # synthetic, per-query-kind
    python benchmarks/run_benchmark.py --models                        # + BGE, cross-encoders
    hybrid-rag download-beir scifact
    python benchmarks/run_benchmark.py --dataset data/beir/scifact --models
    python benchmarks/run_benchmark.py --dataset data/beir/scifact --models --llm --limit 50

Every reranker sees the *same* first-stage candidates (hybrid RRF, cached per query),
so differences are attributable to the reranker alone.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from hybrid_rag import BM25Retriever, DenseRetriever, HashingEmbedder, HybridRetriever
from hybrid_rag.evaluation import evaluate, group_metrics, paired_bootstrap, to_markdown
from hybrid_rag.mock import load_dataset
from hybrid_rag.rerankers import ColBERTReranker, HeuristicReranker, MMRReranker, Reranker

METRICS = ["ndcg@10", "mrr@10", "recall@10", "recall@100", "hit_rate@1"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sample",
                   help="'sample', 'mock', or a BEIR-format directory")
    p.add_argument("--models", action="store_true", help="use neural embedder + rerankers")
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--cross-encoders", nargs="+", default=["minilm", "bge-base"])
    p.add_argument("--llm", action="store_true", help="add Claude listwise reranker (costs $)")
    p.add_argument("--limit", type=int, help="random subset of queries")
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--rerank-depth", type=int, default=30)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    if not ds.queries:
        raise SystemExit(f"{args.dataset!r} has no labelled queries to benchmark")
    if args.limit:
        ds = ds.subset(args.limit)
    print(ds)

    if args.models:
        from hybrid_rag import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.embedder)
        dense_name = f"dense ({args.embedder.split('/')[-1]})"
    else:
        embedder = HashingEmbedder()
        dense_name = "dense (hashing)"

    t = time.perf_counter()
    bm25 = BM25Retriever().index(ds.corpus)
    dense = DenseRetriever(embedder).index(ds.corpus)
    print(f"indexed {len(ds.corpus)} docs in {time.perf_counter() - t:.1f}s")

    def hybrid(fusion: str, **kw) -> HybridRetriever:
        h = HybridRetriever({"bm25": bm25, "dense": dense}, fusion=fusion,
                            candidate_k=args.candidates, **kw)
        h.documents = ds.corpus  # sub-retrievers are already indexed
        return h

    rrf = hybrid("rrf")
    k = args.candidates
    systems: dict[str, Callable] = {
        "bm25": lambda q: bm25.search(q, k),
        dense_name: lambda q: dense.search(q, k),
        "hybrid rrf": lambda q: rrf.search(q, k),
        "hybrid weighted 0.3/0.7": (lambda h: lambda q: h.search(q, k))(
            hybrid("weighted", weights={"bm25": 0.3, "dense": 0.7})),
        "hybrid dbsf": (lambda h: lambda q: h.search(q, k))(hybrid("dbsf")),
        "hybrid combmnz": (lambda h: lambda q: h.search(q, k))(hybrid("combmnz")),
        "hybrid dynamic-alpha": (lambda h: lambda q: h.search(q, k))(
            hybrid("weighted", weights={"bm25": 0.4, "dense": 0.6}, dynamic_alpha=True)),
    }

    cache: dict[str, list] = {}

    def candidates(q: str) -> list:
        if q not in cache:
            cache[q] = rrf.search(q, k)
        return cache[q]

    def reranked(r: Reranker) -> Callable:
        def run(q: str):
            c = candidates(q)
            head = r.rerank(q, c[: args.rerank_depth])
            seen = {x.id for x in head}
            return head + [x for x in c if x.id not in seen]  # keep recall@100 comparable
        return run

    rerankers: dict[str, Reranker] = {
        "heuristic": HeuristicReranker(),
        "colbert (hashing)": ColBERTReranker(),
    }
    if args.models:
        from hybrid_rag.rerankers import CrossEncoderReranker

        for name in args.cross_encoders:
            rerankers[f"cross-encoder {name}"] = CrossEncoderReranker(name)
        first_ce = rerankers[f"cross-encoder {args.cross_encoders[0]}"]
        rerankers[f"cross-encoder {args.cross_encoders[0]} + mmr"] = _Chain(
            first_ce, MMRReranker(embedder, lambda_=0.85))
    if args.llm:
        from hybrid_rag.rerankers import LLMListwiseReranker

        rerankers["claude listwise"] = LLMListwiseReranker(window_size=20, step=10)

    # warm the candidate cache so first-stage latency isn't billed to the first reranker
    for q in ds.queries.values():
        candidates(q)
    for name, r in rerankers.items():
        systems[f"hybrid rrf + {name}"] = reranked(r)

    reports = []
    for name, fn in systems.items():
        t = time.perf_counter()
        reports.append(evaluate(fn, ds, METRICS, name))
        print(f"  {name:<40} {time.perf_counter() - t:6.1f}s")

    base = next(r for r in reports if r.system == "hybrid rrf")
    table = to_markdown(reports)
    sig = "\n".join(
        f"- {r.system}: p(no nDCG@10 gain over hybrid rrf) = "
        f"{paired_bootstrap(base, r, 'ndcg@10', n=5000):.3f}"
        for r in reports if r.system.startswith("hybrid rrf +")
    )
    print("\n" + table + "\n\n" + sig)

    breakdown = ""
    if ds.query_types:
        kinds = sorted(set(ds.query_types.values()))
        shown = [r for r in reports if r.system in (
            "bm25", dense_name, "hybrid rrf", "hybrid dynamic-alpha")
            or r.system.startswith("hybrid rrf + cross-encoder")]
        rows = ["| system | " + " | ".join(f"{k} nDCG@10" for k in kinds) + " |",
                "|---|" + "---:|" * len(kinds)]
        for r in shown:
            g = group_metrics(r, ds.query_types)
            cells = " | ".join(f"{g[k]['ndcg@10']:.3f}" for k in kinds)
            rows.append(f"| {r.system} | {cells} |")
        breakdown = "\nnDCG@10 by query kind:\n\n" + "\n".join(rows) + "\n"
        print(breakdown)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"{ds.name}{'-models' if args.models else ''}{'-llm' if args.llm else ''}"
    (out / f"{stem}.md").write_text(
        f"# {ds.name} ({len(ds.corpus)} docs, {len(ds.queries)} queries)\n\n"
        f"Rerank depth: top {args.rerank_depth} of hybrid RRF candidates.\n\n"
        f"{table}\n{breakdown}\nPaired bootstrap vs hybrid rrf:\n{sig}\n")
    (out / f"{stem}.json").write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    print(f"\nwrote {out / stem}.md/.json")


class _Chain(Reranker):
    name = "chain"

    def __init__(self, first: Reranker, second: Reranker):
        super().__init__()
        self.first, self.second = first, second

    def score(self, query, documents):  # pragma: no cover
        raise NotImplementedError

    def rerank(self, query, results, top_k=None):
        head = self.first.rerank(query, results, 20)
        top = self.second.rerank(query, head, 10)
        seen = {r.id for r in top}
        return top + [r for r in head if r.id not in seen]


if __name__ == "__main__":
    main()
