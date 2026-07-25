"""Run retrieval systems over a labelled dataset and compare them."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import SearchResult
from .datasets import RetrievalDataset
from .metrics import METRICS

SearchFn = Callable[[str], Sequence[SearchResult]]
DEFAULT_METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@20", "hit_rate@1")


def _parse(metric: str) -> tuple[str, int]:
    name, _, k = metric.partition("@")
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}")
    return name, int(k or 10)


def _doc_ids(results: Sequence[SearchResult]) -> list[str]:
    """Map chunks back to their parent document (first occurrence wins) so chunked
    indexes are scored against document-level qrels."""
    seen, out = set(), []
    for r in results:
        did = str(r.document.metadata.get("parent_id", r.id))
        if did not in seen:
            seen.add(did)
            out.append(did)
    return out


@dataclass
class EvalReport:
    system: str
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"system": self.system, "metrics": self.metrics, "latency_ms": self.latency_ms}


def evaluate(search: SearchFn, dataset: RetrievalDataset,
             metrics: Sequence[str] = DEFAULT_METRICS, system: str = "system") -> EvalReport:
    parsed = [(m, *_parse(m)) for m in metrics]
    per_query: dict[str, dict[str, float]] = {}
    latencies = []
    for qid, text in dataset.queries.items():
        t = time.perf_counter()
        ranked = _doc_ids(search(text))
        latencies.append((time.perf_counter() - t) * 1000)
        qrels = dataset.qrels[qid]
        per_query[qid] = {label: METRICS[name](ranked, qrels, k) for label, name, k in parsed}
    agg = {label: float(np.mean([pq[label] for pq in per_query.values()])) if per_query else 0.0
           for label, _, _ in parsed}
    lat = np.array(latencies) if latencies else np.zeros(1)
    return EvalReport(system, agg, per_query,
                      {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                       "mean": float(lat.mean())})


def compare(systems: Mapping[str, SearchFn], dataset: RetrievalDataset,
            metrics: Sequence[str] = DEFAULT_METRICS) -> list[EvalReport]:
    return [evaluate(fn, dataset, metrics, name) for name, fn in systems.items()]


def group_metrics(report: EvalReport, groups: Mapping[str, str]) -> dict[str, dict[str, float]]:
    """Mean of each metric per query group (``groups``: query id -> group label), plus the
    group size under ``"n"``. Shows *where* a system wins, e.g. BM25 on exact identifiers
    vs dense on paraphrases."""
    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(groups.values())):
        qids = [q for q in report.per_query if groups.get(q) == label]
        rows = [report.per_query[q] for q in qids]
        out[label] = {m: float(np.mean([r[m] for r in rows])) for m in report.metrics}
        out[label]["n"] = float(len(qids))
    return out


def paired_bootstrap(a: EvalReport, b: EvalReport, metric: str = "ndcg@10",
                     n: int = 10_000, seed: int = 0) -> float:
    """Approximate p-value that system ``b`` is *not* better than ``a`` on ``metric``
    (paired bootstrap over queries). Small samples make small differences noise -
    check this before declaring a winner."""
    qids = sorted(set(a.per_query) & set(b.per_query))
    if not qids:
        return 1.0
    diff = np.array([b.per_query[q][metric] - a.per_query[q][metric] for q in qids])
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float((samples <= 0).mean())


def to_markdown(reports: Sequence[EvalReport], baseline: str | None = None) -> str:
    if not reports:
        return ""
    labels = list(reports[0].metrics)
    header = "| system | " + " | ".join(labels) + " | p50 ms |"
    sep = "|---|" + "---:|" * (len(labels) + 1)
    best = {m: max(r.metrics[m] for r in reports) for m in labels}
    rows = [header, sep]
    for r in reports:
        cells = []
        for m in labels:
            v = f"{r.metrics[m]:.3f}"
            cells.append(f"**{v}**" if abs(r.metrics[m] - best[m]) < 1e-9 else v)
        rows.append(f"| {r.system} | " + " | ".join(cells) + f" | {r.latency_ms['p50']:.1f} |")
    return "\n".join(rows)


def save_reports(reports: Sequence[EvalReport], path: str | Path) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in reports], indent=2))
