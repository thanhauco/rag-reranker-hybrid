"""Command-line interface.

    hybrid-rag search  "pod keeps restarting" --config configs/hybrid_heuristic.yaml
    hybrid-rag answer  "pod keeps restarting" --config configs/hybrid_claude.yaml
    hybrid-rag eval    --config configs/hybrid_crossencoder.yaml --dataset data/beir/scifact
    hybrid-rag serve   --config configs/hybrid_heuristic.yaml --port 8000
    hybrid-rag download-beir scifact
    hybrid-rag mock    --out data/mock               # synthetic corpus + labelled queries
    hybrid-rag ingest  data/mock/docs --query "TMO-1234"   # index your own files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

from .evaluation import evaluate, group_metrics, to_markdown
from .evaluation.datasets import RetrievalDataset
from .factory import build_pipeline, load_config
from .ingest import SUFFIXES, ingest_path
from .mock import (
    PROBLEMS,
    generate_mock_dataset,
    load_dataset,
    write_beir,
    write_markdown,
)

DEFAULT_CONFIG = {
    "retriever": {"type": "hybrid", "fusion": "rrf",
                  "components": {"bm25": {"type": "bm25"}, "dense": {"type": "dense"}}},
    "reranker": {"type": "heuristic"},
    "generator": {"type": "extractive"},
    "pipeline": {"retrieve_k": 50, "rerank_k": 5},
}
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _dataset(arg: str | None) -> RetrievalDataset:
    return load_dataset(arg or "sample")


def _pipeline(args: argparse.Namespace):
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    ds = _dataset(args.dataset)
    return build_pipeline(cfg).index(ds.corpus), ds


def cmd_search(args: argparse.Namespace) -> None:
    pipe, _ = _pipeline(args)
    resp = pipe.search(args.query, rerank_k=args.k)
    if args.json:
        print(json.dumps(resp.to_dict(), indent=2))
        return
    for i, r in enumerate(resp.results, 1):
        stages = ", ".join(f"{k}={v:.3g}" for k, v in r.scores.items())
        print(f"{i:>2}. [{r.id}] {r.document.title or r.text[:60]}\n    {stages}")
    print(f"\n{resp.candidates} candidates | " +
          " ".join(f"{k}={v:.1f}ms" for k, v in resp.timings_ms.items()))


def cmd_answer(args: argparse.Namespace) -> None:
    pipe, _ = _pipeline(args)
    resp = pipe.answer(args.query, rerank_k=args.k)
    if args.json:
        print(json.dumps(resp.to_dict(include_text=False), indent=2))
        return
    assert resp.answer is not None
    print(resp.answer.text, "\n")
    for c in resp.answer.citations:
        print(f"  [{c.doc_id}] \"{c.cited_text.strip()[:120]}\"")


def cmd_eval(args: argparse.Namespace) -> None:
    pipe, ds = _pipeline(args)
    if args.limit:
        ds = ds.subset(args.limit)
    k = max(int(m.split("@")[1]) for m in args.metrics)
    report = evaluate(lambda q: pipe.search(q, rerank_k=k).results, ds, args.metrics,
                      system=Path(args.config).stem if args.config else "default")
    print(f"{ds}\n")
    print(to_markdown([report]))
    if ds.query_types:
        print("\nBy query type:")
        for label, m in group_metrics(report, ds.query_types).items():
            cells = "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n")
            print(f"  {label:<11} n={int(m['n']):<3} {cells}")


def cmd_mock(args: argparse.Namespace) -> None:
    ds = generate_mock_dataset(args.services, args.problems, args.queries, args.seed)
    out = Path(args.out)
    write_beir(ds, out)
    write_markdown(ds, out / "docs")
    kinds = {}
    for t in ds.query_types.values():
        kinds[t] = kinds.get(t, 0) + 1
    print(f"{ds}\n  BEIR files : {out}/{{corpus,queries}}.jsonl, qrels/test.tsv"
          f"\n  Markdown   : {out}/docs/ ({len(ds.corpus)} files, YAML front matter)"
          f"\n  query kinds: {kinds}")


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    pipe = build_pipeline(cfg)
    stats = ingest_path(pipe, args.path)
    print(f"ingested {stats.summary()}")
    for err in stats.errors[:10]:
        print(f"  ! {err}", file=sys.stderr)
    for q in args.query or []:
        resp = pipe.search(q, rerank_k=args.k)
        print(f"\nQ: {q}")
        for i, r in enumerate(resp.results, 1):
            print(f"{i:>2}. [{r.id}] {r.document.title or r.text[:60]}  ({r.score:.3g})")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    pipe, _ = _pipeline(args)
    uvicorn.run(create_app(pipe), host=args.host, port=args.port)


def cmd_download(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{args.name}.zip"
    if not (out / args.name).exists():
        print(f"downloading {BEIR_URL.format(name=args.name)} ...")
        urllib.request.urlretrieve(BEIR_URL.format(name=args.name), zpath)
        with zipfile.ZipFile(zpath) as z:
            root = out.resolve()
            for member in z.namelist():  # refuse path traversal
                if not (root / member).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in archive: {member}")
            z.extractall(out)
        zpath.unlink()
    print(f"ready: {out / args.name}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hybrid-rag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="pipeline YAML (default: hybrid RRF + heuristic)")
        sp.add_argument("--dataset", default="sample",
                        help="'sample', 'mock', a BEIR-format directory, or a folder of files")

    for name, fn in (("search", cmd_search), ("answer", cmd_answer)):
        sp = sub.add_parser(name)
        sp.add_argument("query")
        sp.add_argument("-k", type=int, default=5)
        sp.add_argument("--json", action="store_true")
        common(sp)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("eval")
    common(sp)
    sp.add_argument("--metrics", nargs="+",
                    default=["ndcg@10", "mrr@10", "recall@10", "hit_rate@1"])
    sp.add_argument("--limit", type=int, help="evaluate on a random subset of queries")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("mock", help="generate a synthetic corpus with labelled queries")
    sp.add_argument("--out", default="data/mock")
    sp.add_argument("--services", type=int, default=12)
    sp.add_argument("--problems", type=int, default=len(PROBLEMS), help=f"1-{len(PROBLEMS)}")
    sp.add_argument("--queries", type=int, default=80)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_mock)

    sp = sub.add_parser("ingest", help=f"index files ({', '.join(sorted(SUFFIXES))})")
    sp.add_argument("path")
    sp.add_argument("--config")
    sp.add_argument("--query", action="append", help="run a search after ingesting (repeatable)")
    sp.add_argument("-k", type=int, default=3)
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("serve")
    common(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("download-beir")
    sp.add_argument("name", help="e.g. scifact, nfcorpus, fiqa, arguana, scidocs")
    sp.add_argument("--out", default="data/beir")
    sp.set_defaults(fn=cmd_download)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
