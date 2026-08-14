"""Full RAG with Claude: hybrid retrieval -> cross-encoder -> Claude listwise rerank ->
Claude answer with native citations.

    pip install -e ".[models,llm]"
    export ANTHROPIC_API_KEY=...        # or: ant auth login
    python examples/04_claude_rag.py "all my pods restarted when the database went down"

Uses claude-opus-5-5. Each query makes 1 listwise rerank call (20 passages, effort=low)
and 1 generation call (5 passages, effort=medium).
"""

import sys

from hybrid_rag import (
    BM25Retriever,
    ClaudeGenerator,
    DenseRetriever,
    HybridRetriever,
    RAGPipeline,
)
from hybrid_rag.evaluation import load_sample
from hybrid_rag.llm import ClaudeClient, LLMError
from hybrid_rag.rerankers import CascadeReranker, HeuristicReranker, LLMListwiseReranker

query = sys.argv[1] if len(sys.argv) > 1 else "all my pods restarted when the database went down"

try:
    from hybrid_rag import SentenceTransformerEmbedder
    from hybrid_rag.rerankers import CrossEncoderReranker

    dense = DenseRetriever(SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"))
    first_reranker = CrossEncoderReranker("minilm")
except ImportError:
    print("(sentence-transformers not installed: using offline embedder + heuristic reranker)")
    dense, first_reranker = DenseRetriever(), HeuristicReranker()

reranker = CascadeReranker([
    (first_reranker, 20),
    (LLMListwiseReranker(ClaudeClient(effort="low"), window_size=20, step=10), 5),
])
pipeline = RAGPipeline(
    HybridRetriever({"bm25": BM25Retriever(), "dense": dense}),
    reranker=reranker,
    generator=ClaudeGenerator(ClaudeClient(effort="medium")),
    retrieve_k=50,
    rerank_k=5,
).index(load_sample().corpus)

try:
    resp = pipeline.answer(query)
except LLMError as e:
    sys.exit(f"Claude call failed: {e}")

print(f"Q: {query}\n\n{resp.answer.text}\n")
print("Citations:")
for c in resp.answer.citations:
    print(f"  [{c.doc_id}] \"{c.cited_text.strip()}\"")
print("\nContext passages (final rank <- per-stage ranks):")
for i, r in enumerate(resp.results, 1):
    print(f"  {i}. {r.id:<4} {r.document.title:<45} {r.ranks}")
print("\nTimings:", {k: f"{v:.0f}ms" for k, v in resp.timings_ms.items()})
print("Tokens:", {k: v for k, v in resp.answer.metadata.items() if "tokens" in k})
