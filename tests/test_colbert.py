import numpy as np
import pytest

from hybrid_rag.rerankers import ColBERTReranker, maxsim
from tests.helpers import CANDIDATES


def test_maxsim():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(q, d) == pytest.approx(1.0 + 0.8)
    assert maxsim(q, np.zeros((0, 2))) == 0.0


def test_colbert_reranker_with_cache():
    r = ColBERTReranker()
    r.precompute([c.document for c in CANDIDATES])
    assert len(r._doc_cache) == 3
    out = r.rerank("failing liveness probe restarts container", CANDIDATES)
    assert out[0].id == "d2"
