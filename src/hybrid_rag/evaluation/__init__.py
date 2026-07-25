from .datasets import RetrievalDataset, load_beir, load_sample
from .evaluator import (
    EvalReport,
    compare,
    evaluate,
    group_metrics,
    paired_bootstrap,
    save_reports,
    to_markdown,
)
from .metrics import (
    average_precision,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "RetrievalDataset",
    "load_beir",
    "load_sample",
    "EvalReport",
    "evaluate",
    "compare",
    "group_metrics",
    "paired_bootstrap",
    "to_markdown",
    "save_reports",
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "average_precision",
]
