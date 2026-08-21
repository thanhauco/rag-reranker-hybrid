"""FastAPI service exposing search / answer / indexing over a configured pipeline."""

# No `from __future__ import annotations` here: FastAPI must resolve the request models,
# which live in create_app's closure, at route-registration time.

from typing import Any

from .pipeline import RAGPipeline
from .types import Document


def create_app(pipeline: RAGPipeline):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'hybrid-rag[api]'") from e

    from .llm import LLMError

    class SearchRequest(BaseModel):
        query: str = Field(min_length=1, max_length=4000)
        retrieve_k: int | None = Field(default=None, ge=1, le=1000)
        rerank_k: int | None = Field(default=None, ge=1, le=100)
        include_text: bool = True

    class DocIn(BaseModel):
        id: str
        text: str
        metadata: dict[str, Any] = {}

    class IndexRequest(BaseModel):
        documents: list[DocIn]

    app = FastAPI(title="hybrid-rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "documents": len(pipeline.retriever),
                "reranker": getattr(pipeline.reranker, "name", None)}

    @app.post("/search")
    def search(req: SearchRequest) -> dict[str, Any]:
        resp = pipeline.search(req.query, req.retrieve_k, req.rerank_k)
        return resp.to_dict(req.include_text)

    @app.post("/answer")
    def answer(req: SearchRequest) -> dict[str, Any]:
        if pipeline.generator is None:
            raise HTTPException(400, "no generator configured")
        try:
            resp = pipeline.answer(req.query, retrieve_k=req.retrieve_k, rerank_k=req.rerank_k)
        except LLMError as e:
            raise HTTPException(502, f"generation failed: {e}") from e
        return resp.to_dict(req.include_text)

    @app.post("/index")
    def index(req: IndexRequest) -> dict[str, Any]:
        """Replace the index with the given documents (rebuilds every retriever)."""
        pipeline.index([Document(d.id, d.text, d.metadata) for d in req.documents])
        return {"indexed": len(pipeline.retriever)}

    return app
