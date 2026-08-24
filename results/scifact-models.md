# scifact (5183 docs, 300 queries)

Rerank depth: top 30 of hybrid RRF candidates.

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

Paired bootstrap vs hybrid rrf:
- hybrid rrf + heuristic: p(no nDCG@10 gain over hybrid rrf) = 1.000
- hybrid rrf + colbert (hashing): p(no nDCG@10 gain over hybrid rrf) = 1.000
- hybrid rrf + cross-encoder minilm: p(no nDCG@10 gain over hybrid rrf) = 0.507
- hybrid rrf + cross-encoder bge-base: p(no nDCG@10 gain over hybrid rrf) = 0.021
- hybrid rrf + cross-encoder minilm + mmr: p(no nDCG@10 gain over hybrid rrf) = 0.706
