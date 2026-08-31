# hybrid-rag · Rerankers for RAG & Hybrid Retrieval

A complete, tested toolkit for the retrieval half of RAG: **hybrid search** (BM25 + dense,
five fusion methods) followed by **multi-stage reranking** (heuristic, ColBERT late
interaction, cross-encoders, Claude pointwise/listwise, MMR), a **grounded generator**
with native citations, and an **evaluation harness** that tells you whether each stage
actually helps.

```
query ─┬─ BM25 ───────┐
       │              ├─ fusion (RRF | weighted | DBSF | CombSUM | CombMNZ) ─ top 100
       └─ dense ──────┘
                 ─▶ cross-encoder / ColBERT ─ top 20 ─▶ Claude listwise ─ top 5 ─▶ MMR ─▶ Claude + citations
```

* **Runs offline out of the box** - core needs only `numpy` + `pyyaml`. BM25, a hashing
  embedder, heuristic and ColBERT-style rerankers, the evaluator and API all work with no
  model downloads. Neural models and Claude are optional extras.
* **Explainable rankings** - every result carries its score and rank from every stage:
  `bm25#7 dense#1 rrf#2 cross_encoder#1`.
* **Config-driven** - swap retrievers, fusion and reranker cascades in YAML.
* **Measured** - nDCG/MRR/recall/MAP, BEIR loader, per-stage ablation benchmark and a
  paired-bootstrap significance test.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"            # offline core
pip install -e ".[models]"             # + sentence-transformers (BGE, cross-encoders, ColBERT)
pip install -e ".[llm]"                # + Claude reranker / generator (anthropic SDK)
pip install -e ".[all,dev]"            # everything (incl. faiss-cpu)
```

Python 3.10+. For Claude, set `ANTHROPIC_API_KEY` (or run `ant auth login`).

## Quickstart

```python
from hybrid_rag import BM25Retriever, DenseRetriever, HybridRetriever, RAGPipeline, Document
from hybrid_rag.rerankers import HeuristicReranker

docs = [Document("oom", "OOMKilled with exit code 137 means the memory limit was exceeded.",
                 {"title": "OOMKilled"}), ...]

pipeline = RAGPipeline(
    HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()}, fusion="rrf"),
    reranker=HeuristicReranker(),
    retrieve_k=50, rerank_k=5,
).index(docs)

for r in pipeline.search("pod restarted with exit code 137").results:
    print(r.id, r.score, r.ranks)      # {'bm25': 1, 'dense': 1, 'rrf': 1, 'heuristic': 1}
```

Neural, production-style:

```python
from hybrid_rag import SentenceTransformerEmbedder
from hybrid_rag.rerankers import CascadeReranker, CrossEncoderReranker, LLMListwiseReranker, MMRReranker

hybrid = HybridRetriever({
    "bm25": BM25Retriever(),
    "dense": DenseRetriever(SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")),
}, fusion="rrf", candidate_k=100)

reranker = CascadeReranker([
    (CrossEncoderReranker("bge-base"), 20),   # 100 -> 20, ~100 ms on GPU
    (LLMListwiseReranker(), 5),               # 20 -> 5, Claude RankGPT-style sliding window
])
```

With Claude generating the answer (each claim cited to the exact supporting span):

```python
from hybrid_rag import ClaudeGenerator
resp = RAGPipeline(hybrid, reranker, ClaudeGenerator()).index(docs).answer("why did my pods restart?")
print(resp.answer.text)
for c in resp.answer.citations:
    print(c.doc_id, c.cited_text)
```

## What's inside

| Area | Components |
|---|---|
| **Chunking** | `FixedSizeChunker`, `SentenceChunker`, `RecursiveChunker` (chunks keep `parent_id`) |
| **Lexical** | `BM25Retriever` - numpy inverted index; Okapi, BM25+, BM25L; title field boosting; save/load |
| **Dense** | `DenseRetriever` + `SentenceTransformerEmbedder` (BGE/E5/nomic prefixes auto-applied), `HashingEmbedder` (offline), `CachedEmbedder`; exact numpy or FAISS HNSW index |
| **Fusion** | `reciprocal_rank_fusion`, `weighted_fusion` (minmax/zscore/dbsf), `distribution_based_fusion`, `comb_sum`, `comb_mnz`; query-aware dynamic alpha |
| **Rerankers** | `HeuristicReranker`, `ColBERTReranker` (MaxSim; HF or hashing token encoders), `CrossEncoderReranker` (MiniLM, BGE, BGE-M3, mxbai, Jina presets), `HTTPReranker` (TEI), `LLMPointwiseReranker`, `LLMListwiseReranker` (Claude), `MMRReranker`, `CascadeReranker`, `EnsembleReranker` |
| **Generation** | `ClaudeGenerator` (citations, streaming, refusal fallbacks), `ExtractiveGenerator` (offline) |
| **Ingestion** | `load_documents` / `ingest_path` for `.md .txt .jsonl .json .csv` (YAML front matter, nested metadata, error accounting, duplicate-id guard) |
| **Mock data** | `generate_mock_dataset` - deterministic synthetic KB with graded qrels and four query kinds (exact / exact+service / paraphrase / typo) |
| **Evaluation** | nDCG, MRR, recall, precision, MAP, hit rate; BEIR loader; per-query-kind breakdown; latency p50/p95; paired bootstrap |
| **Serving** | `hybrid-rag` CLI, FastAPI service (`/search`, `/answer`, `/index`, `/health`), Dockerfile |

## Benchmarks

`python benchmarks/run_benchmark.py --dataset data/beir/scifact --models` - every reranker
reranks the **same** top-30 hybrid-RRF candidates, so differences come from the reranker
alone. Measured on an Apple-silicon laptop CPU (no GPU), BEIR SciFact test split (5,183
abstracts, 300 queries):

| system | ndcg@10 | mrr@10 | recall@10 | recall@100 | hit_rate@1 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.648 | 0.611 | 0.780 | 0.869 | 0.520 | 0.1 |
| dense (bge-small-en-v1.5) | 0.713 | 0.682 | 0.836 | 0.942 | 0.607 | 5.8 |
| hybrid rrf | 0.694 | 0.663 | 0.813 | 0.968 | 0.570 | 6.7 |
| hybrid weighted 0.3/0.7 | 0.720 | 0.688 | 0.840 | **0.972** | 0.603 | 6.7 |
| hybrid dbsf | 0.700 | 0.662 | 0.843 | **0.972** | 0.560 | 6.8 |
| hybrid combmnz | 0.707 | 0.677 | 0.824 | 0.968 | 0.590 | 6.7 |
| hybrid dynamic-alpha | **0.727** | **0.694** | 0.848 | 0.968 | **0.610** | 6.8 |
| hybrid rrf + heuristic | 0.656 | 0.614 | 0.810 | 0.968 | 0.507 | 3.6 |
| hybrid rrf + colbert (hashing) | 0.591 | 0.551 | 0.741 | 0.968 | 0.457 | 3.2 |
| hybrid rrf + cross-encoder minilm | 0.694 | 0.660 | 0.825 | 0.968 | 0.577 | 171.3 |
| hybrid rrf + cross-encoder bge-base | 0.721 | 0.685 | **0.849** | 0.968 | 0.597 | 1175.9 |
| hybrid rrf + cross-encoder minilm + mmr | 0.686 | 0.657 | 0.805 | 0.968 | 0.577 | 458.8 |

p50 is per-query latency on CPU; rerankers include scoring 30 candidates. Paired bootstrap
vs. hybrid RRF (p = probability of no nDCG@10 gain): cross-encoder bge-base **0.021**,
MiniLM 0.51, MiniLM + MMR 0.71, heuristic / hashing-ColBERT 1.00.

**What this shows (and why you must benchmark your own data):**

* **Hybrid recall wins where it matters for reranking**: recall@100 rises from 0.869
  (BM25) / 0.942 (dense) to 0.968-0.972 with any fusion - more relevant documents reach
  the reranker.
* **Equal-weight RRF is not automatically best.** On SciFact, BM25 is much weaker than BGE,
  so equal-weight RRF (0.694) lands *below* dense alone (0.713). Leaning on the stronger
  leg - weighted 0.3/0.7 (0.720) or query-aware dynamic alpha (0.727) - fixes it.
* **A strong cross-encoder is a significant gain** over its candidates (bge-base: +0.027
  nDCG@10, p = 0.02), at ~1 s/query on CPU (use a GPU or a TEI server in production).
  The small MS MARCO MiniLM model is not significant on this scientific domain.
* **Cheap rerankers can hurt out of domain.** The heuristic and hashing-ColBERT
  rerankers lose to plain RRF here; they are fast fallbacks / first cascade stages for
  keyword-heavy corpora, not general-purpose quality boosters.
* **MMR trades ranking metrics for diversity** - judge it on answer quality, not nDCG.

Reproduce:

```bash
hybrid-rag download-beir scifact
python benchmarks/run_benchmark.py --dataset data/beir/scifact --models
python benchmarks/run_benchmark.py --dataset data/beir/scifact --models --llm --limit 50   # adds Claude (costs API credits)
```

The bundled 44-document sample set (`hybrid_rag/data/sample`) is a smoke test - it is
small enough that every system scores > 0.94 nDCG@10. Use BEIR or, better, your own
labelled queries for decisions.

## Mock data and ingestion

No dataset handy? Generate one and push it through the same ingestion path real files take:

```bash
hybrid-rag mock --out data/mock                       # 360 docs + 80 labelled queries
#   data/mock/{corpus,queries}.jsonl, qrels/test.tsv   (BEIR format, with query kinds)
#   data/mock/docs/**.md                               (one Markdown file per doc, YAML front matter)

hybrid-rag ingest data/mock/docs --query "TMO-1234"   # load -> chunk -> index -> search
hybrid-rag eval --dataset data/mock --config configs/hybrid_heuristic.yaml
```

```python
from hybrid_rag import RAGPipeline, BM25Retriever, ingest_path, generate_mock_dataset
stats = ingest_path(RAGPipeline(BM25Retriever()), "my_docs/")   # any folder of .md/.txt/.jsonl/.csv
print(stats.summary())     # 360 files, 360 documents, 18,242 words in 0.19s
```

The mock corpus is a fictional engineering knowledge base: 12 services x 10 problems x 3
document kinds (runbook, postmortem, FAQ), each problem with a unique error code and
described in several different phrasings. Queries are chosen so lexical and dense
retrieval fail in *different* places:

| query kind | example | who should win |
|---|---|---|
| `exact` | `DSK-1933` | BM25 (rare identifier) |
| `exact_svc` | `DSK-1933 session-indexer` | BM25 |
| `paraphrase` | `why does billing-api stop answering` | dense (no shared symptom words) |
| `typo` | `bilng-api memory restarted` | dense (BM25 can't match a misspelt token) |

Measured with BGE-small + MiniLM (nDCG@10, 80 queries, `results/mock-models.md`):

| system | exact | exact+svc | paraphrase | typo |
|---|---:|---:|---:|---:|
| BM25 | **1.000** | **1.000** | 0.329 | 0.151 |
| dense (BGE-small) | 0.849 | 0.915 | **0.583** | 0.740 |
| hybrid RRF | 0.954 | 0.951 | 0.531 | 0.492 |
| hybrid RRF + cross-encoder (MiniLM) | **1.000** | **1.000** | 0.540 | 0.718 |

This is the argument for hybrid + rerank in one table: each single retriever has a blind
spot (BM25 collapses on typos, dense loses 15 points on exact identifiers), RRF covers
the identifier gap, and the cross-encoder recovers typos. Paraphrase stays hard for every
system, which is the signal to try a stronger embedder or the Claude listwise reranker.
See [docs/ingestion-and-testing.md](docs/ingestion-and-testing.md).

## CLI

```bash
hybrid-rag search "pod keeps restarting with exit code 137" -k 5
hybrid-rag search "..." --config configs/hybrid_crossencoder.yaml --json
hybrid-rag answer "what does allkeys-lru do" --config configs/hybrid_claude.yaml
hybrid-rag eval   --config configs/hybrid_heuristic.yaml --dataset data/beir/scifact
hybrid-rag serve  --config configs/hybrid_heuristic.yaml --port 8000
hybrid-rag download-beir nfcorpus
hybrid-rag mock   --out data/mock --services 20 --queries 120 --seed 7
hybrid-rag ingest ./my_docs --config configs/hybrid_crossencoder.yaml --query "how do I rotate certs"
```

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "redis eviction policy for caches", "rerank_k": 3}' | jq '.results[].ranks'
```

## Configs

| File | Pipeline | Needs |
|---|---|---|
| `configs/bm25.yaml` | BM25 baseline | - |
| `configs/hybrid_heuristic.yaml` | BM25 + hashing dense, RRF, heuristic reranker | - |
| `configs/hybrid_weighted_dynamic.yaml` | weighted fusion with per-query alpha, ColBERT (hashing) | - |
| `configs/hybrid_crossencoder.yaml` | sentence chunking, BM25 + BGE, RRF, cross-encoder -> MMR | `[models]` |
| `configs/hybrid_claude.yaml` | BM25 + BGE, RRF, cross-encoder -> Claude listwise, Claude answer | `[models,llm]` |

## Examples

| Script | Shows |
|---|---|
| `examples/01_quickstart.py` | offline hybrid + rerank, per-stage rank trail |
| `examples/02_fusion_comparison.py` | all fusion methods side by side on one query |
| `examples/03_neural_rerankers.py` | BGE + cross-encoders + ColBERT, latency and quality |
| `examples/04_claude_rag.py` | full pipeline with Claude listwise rerank and cited answer |
| `examples/05_custom_reranker.py` | write your own reranker; ensembles and cascades |
| `examples/06_ingest_mock_data.py` | mock Markdown KB -> ingest -> per-query-kind comparison |

## Claude integration notes

* Model: `claude-opus-5-5` by default (configurable per component).
* Rerankers use **structured outputs** (JSON schema) - responses always parse; listwise
  permutations are additionally repaired (dedupe, drop invalid ids, append missing).
* **Effort** is the cost/quality knob: rerankers default to `low`, the generator to
  `medium`. Thinking is adaptive (always on for Opus 5.5), so no `thinking` parameter is sent.
* **Server-side refusal fallbacks** are enabled (`fallbacks="default"`, beta
  `server-side-fallback-2026-07-01`); pass `ClaudeClient(use_fallbacks=False)` on
  platforms that don't support it.
* **Failure is non-fatal for reranking**: API errors, refusals and truncation fall back
  to first-stage order with a logged warning.
* Retrieved passages are fenced in `<passage>` tags / document blocks and the system
  prompts tell the model to treat them as data (prompt-injection hardening).
* `tests/test_claude_sdk.py` runs the real `anthropic` SDK against a mock transport, so
  request shapes are verified in CI without API calls.

## Development

```bash
make test        # 92 offline tests, ~3 s
make test-all    # + tests that download small models
make lint
make bench       # offline ablation on the sample set
make mock        # generate data/mock and run the per-query-kind benchmark
```

## Further reading

* [docs/architecture.md](docs/architecture.md) - module map and design decisions
* [docs/hybrid-search.md](docs/hybrid-search.md) - fusion math and practical defaults
* [docs/reranking.md](docs/reranking.md) - reranker families, trade-offs, how to choose
* [docs/ingestion-and-testing.md](docs/ingestion-and-testing.md) - ingestion formats, mock data, test strategy

## License

MIT
