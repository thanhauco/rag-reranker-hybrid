"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Document:
    """A retrievable unit of text (a whole document or a chunk of one)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))


@dataclass
class SearchResult:
    """A document with the score assigned by the most recent stage.

    ``scores`` keeps the per-stage history (e.g. ``{"bm25": 12.3, "dense": 0.81,
    "rrf": 0.032, "cross_encoder": 7.9}``) so every final ranking is explainable.
    ``ranks`` keeps the 1-based rank each stage assigned.
    """

    document: Document
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.document.id

    @property
    def text(self) -> str:
        return self.document.text

    def with_stage(self, stage: str, score: float, rank: int | None = None) -> SearchResult:
        """Return a copy re-scored by ``stage`` (history is preserved)."""
        scores = {**self.scores, stage: float(score)}
        ranks = dict(self.ranks)
        if rank is not None:
            ranks[stage] = rank
        return replace(self, score=float(score), scores=scores, ranks=ranks)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "score": self.score,
            "scores": self.scores,
            "ranks": self.ranks,
            "metadata": self.document.metadata,
        }
        if include_text:
            out["text"] = self.text
        return out


def assign_ranks(results: list[SearchResult], stage: str) -> list[SearchResult]:
    """Sort by score (desc) and record the 1-based rank for ``stage``."""
    ordered = sorted(results, key=lambda r: r.score, reverse=True)
    for i, r in enumerate(ordered, start=1):
        r.ranks[stage] = i
        r.scores.setdefault(stage, r.score)
    return ordered
