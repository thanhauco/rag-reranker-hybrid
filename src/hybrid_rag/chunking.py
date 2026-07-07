"""Chunking strategies. Retrieval quality is bounded by chunk quality - chunk first."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .text import split_sentences
from .types import Document


class Chunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]: ...

    def split(self, docs: Iterable[Document]) -> list[Document]:
        """Split documents into chunks that keep a pointer to their parent."""
        chunks: list[Document] = []
        for doc in docs:
            pieces = self.split_text(doc.text)
            for i, piece in enumerate(pieces):
                chunk_id = doc.id if len(pieces) == 1 else f"{doc.id}#{i}"
                meta = {**doc.metadata, "parent_id": doc.id, "chunk_index": i,
                        "num_chunks": len(pieces)}
                chunks.append(Document(id=chunk_id, text=piece, metadata=meta))
        return chunks


class FixedSizeChunker(Chunker):
    """Sliding window over words. Simple, predictable, and a strong baseline."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size, self.overlap = chunk_size, overlap

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        step = self.chunk_size - self.overlap
        out = []
        for start in range(0, len(words), step):
            out.append(" ".join(words[start : start + self.chunk_size]))
            if start + self.chunk_size >= len(words):
                break
        return out


class SentenceChunker(Chunker):
    """Packs whole sentences up to ``max_words``; overlaps by N trailing sentences."""

    def __init__(self, max_words: int = 150, overlap_sentences: int = 1):
        self.max_words, self.overlap_sentences = max_words, overlap_sentences

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        out: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for sent in sentences:
            n = len(sent.split())
            if cur and cur_len + n > self.max_words:
                out.append(" ".join(cur))
                cur = cur[-self.overlap_sentences :] if self.overlap_sentences else []
                cur_len = sum(len(s.split()) for s in cur)
            cur.append(sent)
            cur_len += n
        if cur:
            out.append(" ".join(cur))
        return out


class RecursiveChunker(Chunker):
    """Splits on the coarsest separator that yields pieces under ``max_words``
    (paragraph -> line -> sentence -> word), then greedily merges small neighbours."""

    def __init__(self, max_words: int = 200,
                 separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")):
        self.max_words, self.separators = max_words, separators

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text.strip(), 0)
        merged: list[str] = []
        for p in pieces:
            if merged and len(merged[-1].split()) + len(p.split()) <= self.max_words:
                merged[-1] = f"{merged[-1]} {p}"
            else:
                merged.append(p)
        return merged

    def _split(self, text: str, level: int) -> list[str]:
        if len(text.split()) <= self.max_words or level >= len(self.separators):
            return [text] if text else []
        sep = self.separators[level]
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) == 1:
            return self._split(text, level + 1)
        out: list[str] = []
        for p in parts:
            out.extend(self._split(p, level + 1))
        return out


def get_chunker(name: str, **kwargs) -> Chunker:
    registry = {"fixed": FixedSizeChunker, "sentence": SentenceChunker,
                "recursive": RecursiveChunker}
    if name not in registry:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(registry)}")
    return registry[name](**kwargs)
