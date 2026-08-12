import pytest

from hybrid_rag.rerankers import LLMListwiseReranker, LLMPointwiseReranker, repair_permutation
from tests.fakes import FakeLLM
from tests.helpers import CANDIDATES, make_results


@pytest.fixture
def fake_llm():
    return FakeLLM()



def test_llm_pointwise(fake_llm):
    r = LLMPointwiseReranker(fake_llm, batch_size=2)
    out = r.rerank("failing liveness probe restarts", CANDIDATES)
    assert out[0].id == "d2"
    assert len(fake_llm.calls) == 2  # 3 docs / batch_size 2
    assert "untrusted" in fake_llm.calls[0][0]


def test_llm_pointwise_degrades_gracefully():
    out = LLMPointwiseReranker(FakeLLM(fail=True)).rerank("q", CANDIDATES)
    assert [r.id for r in out] == ["d0", "d1", "d2"]  # first-stage order preserved


def test_llm_listwise_sliding_window(fake_llm):
    many = make_results(*[f"filler passage number {i}" for i in range(9)],
                    "failing liveness probe restarts container")
    r = LLMListwiseReranker(fake_llm, window_size=4, step=2)
    out = r.rerank("failing liveness probe restarts", many)
    assert out[0].id == "d9"  # bubbled up from the very bottom
    assert len(fake_llm.calls) == 4  # windows: [6:10] [4:8] [2:6] [0:4]


def test_repair_permutation():
    assert repair_permutation([3, 3, 9, "x", 1], 4) == [2, 0, 1, 3]
    assert repair_permutation([], 3) == [0, 1, 2]
