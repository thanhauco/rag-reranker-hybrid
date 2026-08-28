# Hybrid search: why and how

## Why one retriever is not enough

| | Lexical (BM25) | Dense (bi-encoder) |
|---|---|---|
| Matches | exact tokens | meaning |
| Wins on | error codes, IDs, version numbers, names, rare jargon (`ERR_CONN_RESET`, `x509`, `allkeys-lru`) | paraphrase, synonyms, natural-language questions ("container killed for using too much RAM" -> *OOMKilled*) |
| Fails on | vocabulary mismatch | rare tokens it never saw in training, exact phrases, numbers |
| Index cost | tiny inverted index, no GPU | one embedding per chunk; ANN index |
| Out-of-domain | robust | can degrade sharply (see BEIR) |

Their failure modes are largely **uncorrelated**, which is exactly when combining
helps: hybrid recall@100 is usually higher than either leg alone, and recall at the
first stage is the ceiling for everything downstream - a reranker cannot promote a
document that was never retrieved.

## Fusion methods (all in `hybrid_rag.fusion`)

Let `L_i` be result list *i*, `r_i(d)` the 1-based rank of document *d* in it, `s_i(d)`
its raw score and `w_i` a weight.

### Reciprocal Rank Fusion - `reciprocal_rank_fusion`

    RRF(d) = sum_i  w_i / (k + r_i(d))

* Uses ranks only, so BM25 scores (unbounded, query-dependent) and cosine scores
  ([-1, 1]) never have to be calibrated against each other.
* `k = 60` (Cormack et al., 2009) flattens the top: rank 1 vs rank 2 differ by ~1.6%.
  Smaller `k` makes the very top ranks count more.
* Rewards **agreement**: a doc ranked 2nd by both lists beats a doc ranked 1st by one
  and absent from the other.
* **The default.** It needs no tuning and is hard to beat without labelled data.

### Convex combination - `weighted_fusion`

    S(d) = sum_i  w_i * norm(s_i(d))

with `norm` one of `minmax`, `zscore`, `dbsf`. A document missing from list *i* takes
that list's worst normalised score. With two retrievers this is the classic
`alpha * bm25 + (1 - alpha) * dense`. When `alpha` is tuned on labelled queries it often
beats RRF (Bruch et al., 2023, *An Analysis of Fusion Functions for Hybrid Retrieval*);
untuned it is fragile, because min-max normalisation of a top-k list depends on what
happened to be retrieved.

### Distribution-Based Score Fusion - `distribution_based_fusion`

Normalise each list to `[mu - 3 sigma, mu + 3 sigma]` before the weighted sum (the
approach Qdrant ships). Less sensitive than min-max to a single outlier score.

### CombSUM / CombMNZ - `comb_sum`, `comb_mnz`

Fox & Shaw (1994). CombSUM is the unweighted sum of normalised scores; CombMNZ multiplies
it by the number of lists that retrieved the document - an explicit agreement bonus.

### Query-aware weighting - `HybridRetriever(dynamic_alpha=True)`

`query_aware_alpha()` shifts weight toward BM25 when the query contains identifiers
(`SNAKE_CASE`, dotted names, versions, long numbers, quoted phrases) or is very short,
and toward dense retrieval for long natural-language questions. It is a heuristic
starting point; with labels, train a small classifier on query features to predict
the best alpha per query.

## Practical defaults

1. Pull **deep** candidate lists from each leg (`candidate_k=100`); fusion works on
   overlap and RRF ranks beyond the top 10 still carry signal.
2. Start with **RRF, k=60, equal weights** - but check each leg alone too. When one
   leg is much weaker (BM25 on BEIR SciFact: 0.648 vs BGE-small 0.713 nDCG@10), equal
   weights drag the fusion below the strong leg (0.694); weighting toward it (0.720) or
   dynamic alpha (0.727) recovers and exceeds it. See the README benchmark.
3. Only move to weighted fusion once you have a labelled set, and tune `alpha` on a
   held-out split (`hybrid_rag.evaluation.paired_bootstrap` tells you if a gain is real).
4. Index the **title** in both legs (`BM25Retriever(index_fields=("title",))`,
   `DenseRetriever(include_title=True)` - both defaults).
5. For multilingual or code-heavy corpora consider a third leg (e.g. a second embedding
   model, or SPLADE-style learned sparse vectors) - `HybridRetriever` takes any number.
