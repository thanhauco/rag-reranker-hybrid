"""hybrid_rag - hybrid retrieval + reranking for RAG."""

from .chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from .embeddings import CachedEmbedder, HashingEmbedder, SentenceTransformerEmbedder
from .factory import build_pipeline, load_config
from .fusion import (
    comb_mnz,
    comb_sum,
    distribution_based_fusion,
    fuse,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from .generation import Answer, ClaudeGenerator, ExtractiveGenerator
from .ingest import IngestStats, ingest_path, load_documents
from .mock import generate_mock_dataset, load_dataset
from .pipeline import RAGPipeline, RAGResponse
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from .types import Document, SearchResult

__version__ = "0.1.0"

__all__ = [
    "Document",
    "SearchResult",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "CachedEmbedder",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "reciprocal_rank_fusion",
    "weighted_fusion",
    "distribution_based_fusion",
    "comb_sum",
    "comb_mnz",
    "fuse",
    "RAGPipeline",
    "RAGResponse",
    "Answer",
    "ClaudeGenerator",
    "ExtractiveGenerator",
    "build_pipeline",
    "load_documents",
    "ingest_path",
    "IngestStats",
    "generate_mock_dataset",
    "load_dataset",
    "load_config",
]
