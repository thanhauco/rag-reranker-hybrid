"""hybrid_rag - hybrid retrieval + reranking for RAG."""

from .chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from .embeddings import (
    CachedEmbedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from .fusion import (
    comb_mnz,
    comb_sum,
    distribution_based_fusion,
    fuse,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
)
from .types import (
    Document,
    SearchResult,
)

__version__ = "0.1.0"

__all__ = [
    "Document",
    "SearchResult",
    "BM25Retriever",
    "DenseRetriever",
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
]
