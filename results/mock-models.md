# mock (360 docs, 80 queries)

Rerank depth: top 30 of hybrid RRF candidates.

| system | ndcg@10 | mrr@10 | recall@10 | recall@100 | hit_rate@1 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.620 | 0.681 | 0.738 | 0.996 | 0.588 | 0.0 |
| dense (bge-small-en-v1.5) | 0.772 | **0.906** | 0.854 | 0.992 | **0.875** | 5.5 |
| hybrid rrf | 0.732 | 0.870 | 0.896 | **1.000** | 0.800 | 5.9 |
| hybrid weighted 0.3/0.7 | 0.771 | 0.890 | 0.854 | 0.996 | 0.850 | 5.9 |
| hybrid dbsf | 0.716 | 0.882 | 0.833 | 0.996 | 0.838 | 5.8 |
| hybrid combmnz | 0.732 | 0.865 | 0.842 | 0.996 | 0.812 | 5.8 |
| hybrid dynamic-alpha | 0.721 | 0.868 | 0.796 | 0.996 | 0.825 | 5.8 |
| hybrid rrf + heuristic | 0.727 | 0.836 | 0.879 | **1.000** | 0.762 | 0.8 |
| hybrid rrf + colbert (hashing) | 0.754 | 0.831 | 0.871 | **1.000** | 0.775 | 0.2 |
| hybrid rrf + cross-encoder minilm | **0.814** | 0.850 | 0.908 | **1.000** | 0.775 | 43.5 |
| hybrid rrf + cross-encoder bge-base | 0.770 | 0.867 | 0.892 | **1.000** | 0.812 | 246.8 |
| hybrid rrf + cross-encoder minilm + mmr | 0.810 | 0.851 | **0.912** | **1.000** | 0.775 | 86.2 |

nDCG@10 by query kind:

| system | exact nDCG@10 | exact_svc nDCG@10 | paraphrase nDCG@10 | typo nDCG@10 |
|---|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 0.329 | 0.151 |
| dense (bge-small-en-v1.5) | 0.849 | 0.915 | 0.583 | 0.740 |
| hybrid rrf | 0.954 | 0.951 | 0.531 | 0.492 |
| hybrid dynamic-alpha | 0.895 | 1.000 | 0.578 | 0.412 |
| hybrid rrf + cross-encoder minilm | 1.000 | 1.000 | 0.540 | 0.718 |
| hybrid rrf + cross-encoder bge-base | 0.913 | 0.911 | 0.448 | 0.807 |
| hybrid rrf + cross-encoder minilm + mmr | 0.996 | 0.997 | 0.537 | 0.711 |

Paired bootstrap vs hybrid rrf:
- hybrid rrf + heuristic: p(no nDCG@10 gain over hybrid rrf) = 0.701
- hybrid rrf + colbert (hashing): p(no nDCG@10 gain over hybrid rrf) = 0.183
- hybrid rrf + cross-encoder minilm: p(no nDCG@10 gain over hybrid rrf) = 0.000
- hybrid rrf + cross-encoder bge-base: p(no nDCG@10 gain over hybrid rrf) = 0.089
- hybrid rrf + cross-encoder minilm + mmr: p(no nDCG@10 gain over hybrid rrf) = 0.000
