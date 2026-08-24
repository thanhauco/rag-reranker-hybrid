# mock (360 docs, 80 queries)

Rerank depth: top 30 of hybrid RRF candidates.

| system | ndcg@10 | mrr@10 | recall@10 | recall@100 | hit_rate@1 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.620 | 0.681 | 0.738 | 0.996 | 0.588 | 0.0 |
| dense (hashing) | 0.728 | **0.826** | 0.838 | **1.000** | **0.787** | 0.1 |
| hybrid rrf | 0.688 | 0.797 | 0.846 | **1.000** | 0.725 | 0.3 |
| hybrid weighted 0.3/0.7 | 0.702 | 0.809 | 0.821 | **1.000** | 0.750 | 0.4 |
| hybrid dbsf | 0.682 | 0.816 | 0.821 | **1.000** | 0.762 | 0.4 |
| hybrid combmnz | 0.676 | 0.810 | 0.779 | **1.000** | 0.762 | 0.3 |
| hybrid dynamic-alpha | 0.664 | 0.791 | 0.762 | **1.000** | 0.738 | 0.3 |
| hybrid rrf + heuristic | 0.696 | 0.782 | 0.846 | **1.000** | 0.713 | 0.9 |
| hybrid rrf + colbert (hashing) | **0.765** | 0.826 | **0.854** | **1.000** | 0.775 | 0.2 |

nDCG@10 by query kind:

| system | exact nDCG@10 | exact_svc nDCG@10 | paraphrase nDCG@10 | typo nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 0.329 | 0.151 |
| dense (hashing) | 0.998 | 0.918 | 0.243 | 0.752 |
| hybrid rrf | 1.000 | 0.932 | 0.324 | 0.496 |
| hybrid dynamic-alpha | 0.998 | 1.000 | 0.279 | 0.381 |

Paired bootstrap vs hybrid rrf:
- hybrid rrf + heuristic: p(no nDCG@10 gain over hybrid rrf) = 0.184
- hybrid rrf + colbert (hashing): p(no nDCG@10 gain over hybrid rrf) = 0.001
