# sample-kb (44 docs, 30 queries)

Rerank depth: top 30 of hybrid RRF candidates.

| system | ndcg@10 | mrr@10 | recall@10 | recall@100 | hit_rate@1 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.954 | 0.978 | 0.972 | 0.972 | 0.967 | 0.0 |
| dense (bge-small-en-v1.5) | 0.957 | 0.978 | 0.972 | **1.000** | 0.967 | 5.9 |
| hybrid rrf | 0.960 | 0.983 | 0.972 | **1.000** | 0.967 | 5.7 |
| hybrid weighted 0.3/0.7 | 0.969 | 0.983 | 0.972 | **1.000** | 0.967 | 5.7 |
| hybrid dbsf | 0.970 | 0.983 | 0.972 | **1.000** | 0.967 | 5.7 |
| hybrid combmnz | **0.982** | **1.000** | 0.972 | **1.000** | **1.000** | 5.7 |
| hybrid dynamic-alpha | 0.970 | 0.983 | 0.972 | **1.000** | 0.967 | 7.6 |
| hybrid rrf + heuristic | 0.963 | **1.000** | 0.972 | **1.000** | **1.000** | 0.9 |
| hybrid rrf + colbert (hashing) | 0.967 | 0.983 | 0.972 | **1.000** | 0.967 | 0.1 |
| hybrid rrf + cross-encoder minilm | 0.962 | 0.983 | 0.972 | **1.000** | 0.967 | 36.2 |
| hybrid rrf + cross-encoder bge-base | 0.972 | 0.983 | **0.983** | **1.000** | 0.967 | 206.7 |
| hybrid rrf + cross-encoder minilm + mmr | 0.946 | 0.983 | 0.933 | **1.000** | 0.967 | 68.0 |

Paired bootstrap vs hybrid rrf:
- hybrid rrf + heuristic: p(no nDCG@10 gain over hybrid rrf) = 0.499
- hybrid rrf + colbert (hashing): p(no nDCG@10 gain over hybrid rrf) = 0.158
- hybrid rrf + cross-encoder minilm: p(no nDCG@10 gain over hybrid rrf) = 0.425
- hybrid rrf + cross-encoder bge-base: p(no nDCG@10 gain over hybrid rrf) = 0.058
- hybrid rrf + cross-encoder minilm + mmr: p(no nDCG@10 gain over hybrid rrf) = 0.951
