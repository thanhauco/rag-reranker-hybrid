"""Second-stage rerankers. Heavy backends (torch, anthropic) are imported lazily."""

from .base import (
    IdentityReranker,
    Reranker,
)
from .colbert import (
    ColBERTReranker,
    HashingTokenEncoder,
    HFTokenEncoder,
    maxsim,
)
from .heuristic import (
    HeuristicReranker,
)
from .http import (
    HTTPReranker,
)
from .mmr import (
    MMRReranker,
)


def __getattr__(name: str):
    # CrossEncoderReranker pulls in torch at call time only; keep import cheap.
    if name == "CrossEncoderReranker":
        from .cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    raise AttributeError(name)

__all__ = [
    "Reranker",
    "IdentityReranker",
    "HeuristicReranker",
    "CrossEncoderReranker",
    "ColBERTReranker",
    "HashingTokenEncoder",
    "HFTokenEncoder",
    "MMRReranker",
    "HTTPReranker",
    "maxsim",
]
