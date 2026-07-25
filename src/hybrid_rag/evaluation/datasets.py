"""Retrieval datasets: the bundled sample set and any BEIR-format directory.

BEIR layout::

    <dir>/corpus.jsonl     {"_id", "title", "text"}
    <dir>/queries.jsonl    {"_id", "text"}
    <dir>/qrels/test.tsv   query-id <TAB> corpus-id <TAB> score   (with a header row)
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Document

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class RetrievalDataset:
    name: str
    corpus: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    # optional query-id -> category (e.g. "exact", "paraphrase", "typo") for breakdowns
    query_types: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # keep only queries that have judgments
        self.queries = {q: t for q, t in self.queries.items() if self.qrels.get(q)}

    def subset(self, n_queries: int, seed: int = 0) -> RetrievalDataset:
        ids = sorted(self.queries)
        random.Random(seed).shuffle(ids)
        keep = set(ids[:n_queries])
        return RetrievalDataset(self.name, self.corpus,
                                {q: self.queries[q] for q in keep},
                                {q: self.qrels[q] for q in keep},
                                {q: t for q, t in self.query_types.items() if q in keep})

    def __repr__(self) -> str:
        return (f"RetrievalDataset({self.name!r}, docs={len(self.corpus)}, "
                f"queries={len(self.queries)})")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(path: str | Path, split: str = "test", name: str | None = None) -> RetrievalDataset:
    path = Path(path)
    corpus = [
        Document(str(r["_id"]), r.get("text", ""),
                 {"title": r.get("title", ""), **r.get("metadata", {})})
        for r in _read_jsonl(path / "corpus.jsonl")
    ]
    queries = {str(r["_id"]): r["text"] for r in _read_jsonl(path / "queries.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] in ("query-id", "query_id", "qid"):
                continue
            qid, did, score = row[0], row[1], int(float(row[2]))
            qrels.setdefault(qid, {})[did] = score
    return RetrievalDataset(name or path.name, corpus, queries, qrels)


def load_sample() -> RetrievalDataset:
    """The bundled engineering-knowledge-base sample (``hybrid_rag/data/sample``)."""
    return load_beir(DATA_DIR / "sample", name="sample-kb")
