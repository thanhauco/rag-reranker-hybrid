# Reranking guide

A first-stage retriever optimises **recall** cheaply over millions of documents. A
reranker optimises **precision** expensively over the top 20-100 candidates. Splitting
the work this way is what makes high-quality RAG affordable.

```
corpus (10^6)  --BM25 + dense-->  100  --cross-encoder-->  20  --LLM listwise-->  5  --> generator
              ~10 ms                     ~50-200 ms (GPU)          ~1-3 s
```

## Reranker families in this repo

| Reranker | Class | How it scores | Quality | Cost per query | When to use |
|---|---|---|---|---|---|
| Identity | `IdentityReranker` | keeps first-stage order | - | 0 | ablation baseline |
| Heuristic | `HeuristicReranker` | coverage, proximity, phrase, title + prior | domain-dependent (hurt on SciFact) | microseconds | keyword-heavy corpora; no-GPU fallback; first cascade stage |
| ColBERT / late interaction | `ColBERTReranker` | MaxSim over per-token embeddings | ++ with a trained ColBERT encoder; the hashing encoder is a baseline only | 1 query encode + matmuls (passages precomputed) | large candidate sets, latency-bound |
| Cross-encoder | `CrossEncoderReranker` | joint transformer over (query, passage) | +++ | 1 forward pass per candidate | the default second stage |
| Remote cross-encoder | `HTTPReranker` | TEI `/rerank` endpoint | +++ | network + GPU server | serve the model separately from the app |
| LLM pointwise | `LLMPointwiseReranker` | Claude grades each passage 0-10 | +++ | 1 call per batch of 10 | you need absolute scores / thresholds |
| LLM listwise | `LLMListwiseReranker` | Claude orders a window (RankGPT sliding window) | ++++ | ~ceil((n - w)/s) + 1 calls | hard queries, small final k, quality over cost |
| MMR | `MMRReranker` | relevance minus redundancy | (diversity) | embeddings of candidates | after relevance reranking, before generation |
| Cascade | `CascadeReranker` | funnel of the above | - | dominated by last stage | production |
| Ensemble | `EnsembleReranker` | RRF/weighted fusion of several rerankers | + | sum of members | combine a relevance model with business signals |

### Cross-encoders

`CrossEncoderReranker(model_name)` accepts any `sentence-transformers` CrossEncoder or a
preset: `minilm` (fast, English), `minilm-l12`, `bge-base`, `bge-large`, `bge-m3`
(multilingual, long context), `mxbai-base`, `jina-v2`. Scores are passed through a
sigmoid so `min_score` thresholds are comparable across queries.

Rules of thumb:

* Rerank depth 20-50 captures most of the gain; beyond ~100 latency grows linearly
  while nDCG@10 barely moves.
* Include the title in the passage (`include_title=True`, default).
* Cross-encoders are trained on passage-length inputs (MS MARCO ~ 60 words). Chunk
  documents before reranking rather than truncating them at 512 tokens.

### Late interaction (ColBERT)

`ColBERTReranker` computes `sum_i max_j q_i . d_j`. Passage token matrices are cached
(`precompute()` is called automatically by `RAGPipeline.index`). The bundled
`HFTokenEncoder` uses any Hugging Face encoder's token states; for production ColBERT
(trained projection head, query augmentation, PLAID compression) plug PyLate or
RAGatouille in through the two-method `TokenEncoder` protocol. `HashingTokenEncoder`
gives an offline, typo-tolerant soft-matching baseline.

### LLM rerankers (Claude)

Both use structured outputs (JSON schema) so the response is always parseable, enable
server-side refusal fallbacks, and **degrade gracefully** - on an API error, refusal or
truncation the affected candidates keep their first-stage order and a warning is logged.

* **Pointwise** batches 10 passages per call and runs batches concurrently. Grades are
  absolute (0-10), so `min_score=5` drops passages the model judged unhelpful - a
  useful guard against feeding irrelevant context to the generator.
* **Listwise** implements the RankGPT sliding window: window `w=20`, step `s=10`,
  walking from the bottom of the list to the top so a strong passage at rank 50 can
  bubble to rank 1 in one pass. The model returns a permutation; invalid or duplicate
  ids are repaired (`repair_permutation`).
* Passages are wrapped in `<passage>` tags and the system prompt instructs the model
  to treat them as data, which mitigates prompt injection from retrieved content.
* Default model `claude-opus-5-5` at `effort="low"`: relevance judgement rarely needs
  deep deliberation. Raise effort if your eval shows headroom; measure cost per query
  with `response.usage`.

Use the LLM as the **last** stage over at most ~20 candidates.

### MMR

`MMRReranker(lambda_=0.7)` greedily picks the passage maximising
`lambda * relevance - (1 - lambda) * max_similarity_to_already_picked`. Use it right
before generation when chunks overlap (same page split into several chunks, mirrored
docs). Expect nDCG to drop slightly - it optimises coverage of the context window,
not ranking metrics - so evaluate it on answer quality.

## Choosing a setup

| Constraint | Recommended |
|---|---|
| CPU only, < 20 ms budget | hybrid RRF -> `HeuristicReranker` or `ColBERTReranker` |
| One GPU, < 200 ms | hybrid RRF (top 100) -> `CrossEncoderReranker("bge-base")` on top 30 |
| Multilingual | `bge-m3` embedder + `CrossEncoderReranker("bge-m3")` |
| Quality first, seconds OK | hybrid -> cross-encoder top 50 -> 20 -> `LLMListwiseReranker` -> 5 |
| Must not answer from weak context | `LLMPointwiseReranker(min_score=5)` before generation |

Always confirm with your own labelled queries: `python benchmarks/run_benchmark.py`
reports every stage on the same candidates plus a paired-bootstrap significance test.
