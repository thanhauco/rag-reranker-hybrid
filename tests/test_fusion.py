import numpy as np
import pytest

from hybrid_rag import Document, SearchResult
from hybrid_rag.fusion import (
    comb_mnz,
    dbsf,
    fuse,
    minmax,
    reciprocal_rank_fusion,
    weighted_fusion,
    zscore,
)


def _list(name, pairs):
    return [SearchResult(Document(d, d), s, {name: s}, {name: i + 1})
            for i, (d, s) in enumerate(pairs)]


@pytest.fixture
def lists():
    return {
        "bm25": _list("bm25", [("a", 12.0), ("b", 8.0), ("c", 1.0)]),
        "dense": _list("dense", [("b", 0.91), ("d", 0.90), ("a", 0.10)]),
    }


def test_rrf_formula(lists):
    out = {r.id: r.score for r in reciprocal_rank_fusion(lists, k=60)}
    assert out["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["d"] == pytest.approx(1 / 62)
    assert max(out, key=out.get) == "b"  # agreement beats a single first place


def test_rrf_weights(lists):
    out = reciprocal_rank_fusion(lists, weights={"bm25": 5.0, "dense": 1.0})
    assert out[0].id == "a"


def test_rrf_keeps_stage_history(lists):
    b = next(r for r in reciprocal_rank_fusion(lists) if r.id == "b")
    assert b.scores["bm25"] == 8.0 and b.scores["dense"] == 0.91
    assert b.ranks == {"bm25": 2, "dense": 1, "rrf": 1}


def test_weighted_is_scale_invariant(lists):
    scaled = {"bm25": _list("bm25", [("a", 1200.0), ("b", 800.0), ("c", 100.0)]),
              "dense": lists["dense"]}
    a = [r.id for r in weighted_fusion(lists)]
    b = [r.id for r in weighted_fusion(scaled)]
    assert a == b


def test_weighted_requires_all_weights(lists):
    with pytest.raises(ValueError):
        weighted_fusion(lists, weights={"bm25": 1.0})


def test_combmnz_rewards_agreement(lists):
    out = comb_mnz(lists)
    assert out[0].id in {"a", "b"}
    d = next(r for r in out if r.id == "d")
    assert d.score <= out[0].score


def test_normalisers():
    s = np.array([1.0, 2.0, 3.0])
    assert minmax(s).tolist() == [0.0, 0.5, 1.0]
    assert zscore(s).mean() == pytest.approx(0.0)
    assert ((dbsf(s) >= 0) & (dbsf(s) <= 1)).all()
    assert minmax(np.array([5.0, 5.0])).tolist() == [1.0, 1.0]


def test_fuse_dispatch_and_top_k(lists):
    assert len(fuse(lists, "rrf", top_k=2)) == 2
    with pytest.raises(ValueError):
        fuse(lists, "nope")


def test_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion({"a": [], "b": []}) == []
    assert weighted_fusion({"a": []}) == []


def test_weighted_zscore_missing_doc_is_not_rewarded(lists):
    # "d" appears only in the dense list; under z-scores a miss in bm25 must count as the
    # worst bm25 score, not the mean (0).
    out = {r.id: r.score for r in weighted_fusion(lists, normalization="zscore")}
    bm25_z = zscore(np.array([12.0, 8.0, 1.0]))
    dense_z = zscore(np.array([0.91, 0.90, 0.10]))
    assert out["d"] == pytest.approx(bm25_z.min() + dense_z[1])
