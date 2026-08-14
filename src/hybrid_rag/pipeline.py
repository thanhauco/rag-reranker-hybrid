"""End-to-end RAG pipeline: retrieve -> (fuse) -> rerank -> generate, with timings."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .chunking import Chunker
from .generation import Answer, Generator
from .rerankers.base import Reranker
from .retrievers.base import Retriever
from .types import Document, SearchResult


@dataclass
class RAGResponse:
    query: str
    results: list[SearchResult]
    answer: Answer | None = None
    candidates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer.to_dict() if self.answer else None,
            "results": [r.to_dict(include_text) for r in self.results],
            "candidates": self.candidates,
            "timings_ms": self.timings_ms,
        }


class RAGPipeline:
    """
    retrieve_k : candidates pulled from the first stage (recall budget)
    rerank_k   : passages kept after reranking and handed to the generator (precision)
    """

    def __init__(self, retriever: Retriever, reranker: Reranker | None = None,
                 generator: Generator | None = None, chunker: Chunker | None = None,
                 retrieve_k: int = 50, rerank_k: int = 5):
        self.retriever, self.reranker, self.generator = retriever, reranker, generator
        self.chunker = chunker
        self.retrieve_k, self.rerank_k = retrieve_k, rerank_k

    def index(self, documents: Iterable[Document]) -> RAGPipeline:
        docs = list(documents)
        if self.chunker is not None:
            docs = self.chunker.split(docs)
        self.retriever.index(docs)
        precompute = getattr(self.reranker, "precompute", None)
        if callable(precompute):
            precompute(docs)
        return self

    def search(self, query: str, retrieve_k: int | None = None,
               rerank_k: int | None = None) -> RAGResponse:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        candidates = self.retriever.search(query, retrieve_k or self.retrieve_k)
        timings["retrieve"] = (time.perf_counter() - t0) * 1000
        k = rerank_k or self.rerank_k
        if self.reranker is not None and candidates:
            t1 = time.perf_counter()
            results = self.reranker.rerank(query, candidates, k)
            timings["rerank"] = (time.perf_counter() - t1) * 1000
        else:
            results = candidates[:k]
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RAGResponse(query, results, candidates=len(candidates), timings_ms=timings)

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        if self.generator is None:
            raise ValueError("pipeline has no generator configured")
        resp = self.search(query, **kwargs)
        t = time.perf_counter()
        resp.answer = self.generator.generate(query, resp.results)
        resp.timings_ms["generate"] = (time.perf_counter() - t) * 1000
        resp.timings_ms["total"] += resp.timings_ms["generate"]
        return resp
