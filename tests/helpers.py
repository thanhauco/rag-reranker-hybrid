"""Shared test data for reranker tests."""

from hybrid_rag import Document, SearchResult


def make_results(*texts):
    return [SearchResult(Document(f"d{i}", t), float(len(texts) - i))
            for i, t in enumerate(texts)]


CANDIDATES = make_results(
    "liveness liveness liveness liveness",                         # keyword stuffing
    "Kubernetes probes have fields like periodSeconds.",
    "A failing liveness probe restarts the container; readiness controls traffic.",
)
QUERY = "what does a failing liveness probe do"
