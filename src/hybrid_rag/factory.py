"""Build pipelines from plain dicts / YAML so experiments are config changes, not code.

See ``configs/*.yaml`` for complete examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .chunking import Chunker, get_chunker
from .embeddings import CachedEmbedder, Embedder, get_embedder
from .generation import ClaudeGenerator, ExtractiveGenerator, Generator
from .llm import ClaudeClient
from .pipeline import RAGPipeline
from .rerankers import (
    CascadeReranker,
    ColBERTReranker,
    EnsembleReranker,
    HeuristicReranker,
    HFTokenEncoder,
    HTTPReranker,
    IdentityReranker,
    LLMListwiseReranker,
    LLMPointwiseReranker,
    MMRReranker,
    Reranker,
)
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever


def _split(cfg: dict[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if isinstance(cfg, str):
        return cfg, {}
    cfg = dict(cfg)
    return cfg.pop("type"), cfg


def build_embedder(cfg: dict[str, Any] | str | None) -> Embedder:
    if cfg is None:
        return get_embedder("hashing")
    kind, kw = _split(cfg)
    cache = kw.pop("cache", False)
    emb = get_embedder(kind, **kw)
    return CachedEmbedder(emb) if cache else emb


def build_retriever(cfg: dict[str, Any] | str) -> Retriever:
    kind, kw = _split(cfg)
    if kind == "bm25":
        return BM25Retriever(**kw)
    if kind == "dense":
        return DenseRetriever(build_embedder(kw.pop("embedder", None)), **kw)
    if kind == "hybrid":
        comps = {name: build_retriever(c) for name, c in kw.pop("components").items()}
        return HybridRetriever(comps, **kw)
    raise ValueError(f"unknown retriever {kind!r}")


def _llm_client(kw: dict[str, Any]) -> ClaudeClient:
    keys = ("model", "effort", "max_tokens", "use_fallbacks")
    return ClaudeClient(**{k: kw.pop(k) for k in keys if k in kw})


def build_reranker(cfg: dict[str, Any] | str | None) -> Reranker | None:
    if cfg is None:
        return None
    kind, kw = _split(cfg)
    kw.pop("keep", None)
    if kind == "identity":
        return IdentityReranker(**kw)
    if kind == "heuristic":
        return HeuristicReranker(**kw)
    if kind == "cross_encoder":
        from .rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(**kw)
    if kind == "colbert":
        enc = kw.pop("encoder", None)
        encoder = HFTokenEncoder(**enc) if isinstance(enc, dict) else None
        return ColBERTReranker(encoder, **kw)
    if kind == "llm_pointwise":
        return LLMPointwiseReranker(_llm_client(kw), **kw)
    if kind == "llm_listwise":
        return LLMListwiseReranker(_llm_client(kw), **kw)
    if kind == "mmr":
        return MMRReranker(build_embedder(kw.pop("embedder", None)), **kw)
    if kind == "http":
        return HTTPReranker(**kw)
    if kind == "cascade":
        stages = [(build_reranker(s), (s.get("keep") if isinstance(s, dict) else None))
                  for s in kw.pop("stages")]
        return CascadeReranker(stages)  # type: ignore[arg-type]
    if kind == "ensemble":
        members = {name: build_reranker(c) for name, c in kw.pop("members").items()}
        return EnsembleReranker(members, **kw)  # type: ignore[arg-type]
    raise ValueError(f"unknown reranker {kind!r}")


def build_generator(cfg: dict[str, Any] | str | None) -> Generator | None:
    if cfg is None:
        return None
    kind, kw = _split(cfg)
    if kind == "extractive":
        return ExtractiveGenerator(**kw)
    if kind == "claude":
        return ClaudeGenerator(_llm_client(kw), **kw)
    raise ValueError(f"unknown generator {kind!r}")


def build_chunker(cfg: dict[str, Any] | str | None) -> Chunker | None:
    if cfg is None:
        return None
    kind, kw = _split(cfg)
    return get_chunker(kind, **kw)


def build_pipeline(cfg: dict[str, Any]) -> RAGPipeline:
    return RAGPipeline(
        retriever=build_retriever(cfg["retriever"]),
        reranker=build_reranker(cfg.get("reranker")),
        generator=build_generator(cfg.get("generator")),
        chunker=build_chunker(cfg.get("chunker")),
        **cfg.get("pipeline", {}),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())
