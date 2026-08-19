"""Load real-world files into :class:`~hybrid_rag.types.Document` objects.

Supported inputs (a single file, or a directory walked recursively in sorted order):

=============  ===============================================================
``.md .txt``   one document per file. Optional YAML front matter (``---``) becomes
               metadata; ``title``/``id`` keys are honoured. Without a title, the
               first ``# heading`` (else the file name) is used.
``.jsonl``     one document per line: ``{"id", "text", "title", ...}``; every other
               key (and a nested ``"metadata"`` object) becomes metadata.
``.json``      a list of such objects.
``.csv``       header row with at least a ``text`` column (plus optional ``id``,
               ``title``; other columns become metadata).
=============  ===============================================================

Bad input is never silently half-ingested: unreadable files and malformed records are
counted in :class:`IngestStats` (and logged), empty documents are skipped, and duplicate
ids raise, because two documents sharing an id would silently overwrite each other in
every downstream index and evaluation.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import Document

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUFFIXES = TEXT_SUFFIXES | {".jsonl", ".json", ".csv"}
MAX_BYTES = 10 * 1024 * 1024
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*(?:\n|\Z)", re.S)
_HEADING = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.M)


@dataclass
class IngestStats:
    files: int = 0
    documents: int = 0
    chunks: int = 0
    words: int = 0
    skipped_empty: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        parts = [f"{self.files} files", f"{self.documents} documents"]
        if self.chunks and self.chunks != self.documents:
            parts.append(f"{self.chunks} chunks")
        parts.append(f"{self.words:,} words")
        if self.skipped_empty:
            parts.append(f"{self.skipped_empty} empty skipped")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        return ", ".join(parts) + f" in {self.seconds:.2f}s"


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for p in sorted(path.rglob("*")):
        rel = p.relative_to(path)
        if p.is_file() and p.suffix.lower() in SUFFIXES and not any(
                part.startswith(".") for part in rel.parts):
            yield p


def _doc_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root) if root != path else Path(path.name)
    return rel.with_suffix("").as_posix()


def split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML front matter: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError("front matter must be a mapping")
    return meta, raw[m.end():]


def _parse_text_file(path: Path, root: Path) -> list[Document]:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    title = str(meta.pop("title", "") or "")
    doc_id = str(meta.pop("id", "") or _doc_id(path, root))
    if not title:
        m = _HEADING.search(body)
        if m:
            title = m.group(1)
            body = body[:m.start()] + body[m.end():]
        else:
            title = path.stem.replace("_", " ").replace("-", " ")
    elif (m := _HEADING.match(body.lstrip())) and m.group(1) == title:
        body = body.lstrip()[m.end():]
    return [Document(doc_id, body.strip(), {**meta, "title": title,
                                            "source": _doc_id(path, root) + path.suffix})]


def _record(rec: dict[str, Any], fallback_id: str, source: str) -> Document:
    if not isinstance(rec, dict):
        raise ValueError("record is not an object")
    rec = dict(rec)
    text = rec.pop("text", None)
    if text is None:
        raise ValueError("record has no 'text'")
    doc_id = str(rec.pop("id", None) or rec.pop("_id", None) or fallback_id)
    title = str(rec.pop("title", "") or "")
    nested = rec.pop("metadata", None)
    meta = {**(nested if isinstance(nested, dict) else {}), **rec}
    return Document(doc_id, str(text).strip(), {**meta, "title": title, "source": source})


def _parse_jsonl(path: Path, root: Path, stats: IngestStats) -> list[Document]:
    docs, stem = [], _doc_id(path, root)
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                docs.append(_record(json.loads(line), f"{stem}:{n}", f"{stem}.jsonl:{n}"))
            except (ValueError, json.JSONDecodeError) as e:
                stats.errors.append(f"{path.name}:{n}: {e}")
    return docs


def _parse_json(path: Path, root: Path, stats: IngestStats) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("top-level JSON value must be a list of objects")
    docs, stem = [], _doc_id(path, root)
    for n, rec in enumerate(data):
        try:
            docs.append(_record(rec, f"{stem}:{n}", f"{stem}.json[{n}]"))
        except ValueError as e:
            stats.errors.append(f"{path.name}[{n}]: {e}")
    return docs


def _parse_csv(path: Path, root: Path, stats: IngestStats) -> list[Document]:
    docs, stem = [], _doc_id(path, root)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV needs a header row with a 'text' column")
        for n, row in enumerate(reader, start=1):
            try:
                docs.append(_record({k: v for k, v in row.items() if k and v not in (None, "")},
                                    f"{stem}:{n}", f"{stem}.csv:{n}"))
            except ValueError as e:
                stats.errors.append(f"{path.name}:{n}: {e}")
    return docs


def load_documents(path: str | Path, stats: IngestStats | None = None) -> list[Document]:
    """Read every supported file under ``path``. Pass ``stats`` to collect counts/errors."""
    stats = stats if stats is not None else IngestStats()
    t0 = time.perf_counter()
    root = base = Path(path)
    docs: list[Document] = []
    seen: dict[str, str] = {}
    for f in _iter_files(root):
        try:
            if f.stat().st_size > MAX_BYTES:
                raise ValueError(f"file larger than {MAX_BYTES // 2**20} MB")
            suffix = f.suffix.lower()
            if suffix in TEXT_SUFFIXES:
                parsed = _parse_text_file(f, base)
            elif suffix == ".jsonl":
                parsed = _parse_jsonl(f, base, stats)
            elif suffix == ".json":
                parsed = _parse_json(f, base, stats)
            else:
                parsed = _parse_csv(f, base, stats)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as e:
            stats.errors.append(f"{f.name}: {e}")
            log.warning("skipping %s: %s", f, e)
            continue
        stats.files += 1
        for d in parsed:
            if not d.text:
                stats.skipped_empty += 1
                continue
            if d.id in seen:
                raise ValueError(f"duplicate document id {d.id!r} "
                                 f"({seen[d.id]} and {d.metadata.get('source')})")
            seen[d.id] = str(d.metadata.get("source"))
            docs.append(d)
            stats.words += len(d.text.split())
    stats.documents = len(docs)
    stats.seconds = time.perf_counter() - t0
    return docs


def ingest_path(pipeline, path: str | Path) -> IngestStats:
    """Load ``path`` and (re)build ``pipeline``'s index from it. Chunking, if the pipeline
    has a chunker, happens inside ``pipeline.index``."""
    stats = IngestStats()
    docs = load_documents(path, stats)
    if not docs:
        raise ValueError(f"no documents found under {path} "
                         f"({'; '.join(stats.errors[:3]) or 'no supported files'})")
    t0 = time.perf_counter()
    pipeline.index(docs)
    stats.seconds += time.perf_counter() - t0
    stats.chunks = len(pipeline.retriever)
    return stats
