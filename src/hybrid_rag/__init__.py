"""hybrid_rag - hybrid retrieval + reranking for RAG."""

from .chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from .retrievers import (
    BM25Retriever,
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
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
]
