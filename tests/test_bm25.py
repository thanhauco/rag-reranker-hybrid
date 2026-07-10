import math

import numpy as np
import pytest

from hybrid_rag import BM25Retriever, Document


def test_bm25_exact_match_ranks_first(docs):
    r = BM25Retriever().index(docs)
    res = r.search("too many clients", k=3)
    assert res[0].id == "b"
    assert res[0].ranks["bm25"] == 1


def test_bm25_no_overlap_returns_nothing(docs):
    assert BM25Retriever().index(docs).search("zebra quantum") == []


def test_bm25_matches_reference_formula():
    docs = [Document(str(i), t) for i, t in enumerate(["apple banana", "apple apple cherry",
                                                        "banana cherry durian"])]
    r = BM25Retriever(k1=1.5, b=0.75).index(docs)
    s = r.get_scores("apple")
    # df=2 of 3 -> raw Okapi IDF is negative and must be floored to a small positive value
    assert math.log((3 - 2 + 0.5) / (2 + 0.5)) < 0
    idf = r._idf["apple"]
    assert idf > 0
    avgdl = np.mean([2, 3, 3])
    tf, dl = 2, 3
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert s[1] == pytest.approx(expected, rel=1e-5)
    assert s[2] == 0


@pytest.mark.parametrize("variant", ["okapi", "plus", "l"])
def test_bm25_variants(docs, variant):
    res = BM25Retriever(variant=variant).index(docs).search("fox dog", k=5)
    assert {r.id for r in res[:2]} == {"a", "d"}


def test_bm25_save_load(tmp_path, docs):
    r = BM25Retriever().index(docs)
    r.save(tmp_path / "bm25.pkl")
    r2 = BM25Retriever.load(tmp_path / "bm25.pkl")
    assert [x.id for x in r2.search("fox")] == [x.id for x in r.search("fox")]
