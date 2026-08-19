import json

import pytest

from hybrid_rag import BM25Retriever, DenseRetriever, HybridRetriever
from hybrid_rag.evaluation import evaluate, group_metrics, load_beir
from hybrid_rag.mock import (
    PROBLEMS,
    QUERY_KINDS,
    generate_mock_dataset,
    load_dataset,
    write_beir,
)


@pytest.fixture(scope="module")
def mock():
    return generate_mock_dataset()


def test_shape_and_determinism(mock):
    assert len(mock.corpus) == 12 * len(PROBLEMS) * 3
    assert len(mock.queries) == 80
    again = generate_mock_dataset()
    assert [d.text for d in again.corpus] == [d.text for d in mock.corpus]
    assert again.queries == mock.queries
    other = generate_mock_dataset(seed=1)
    assert other.queries != mock.queries


def test_error_codes_unique_and_present_in_docs(mock):
    codes = {d.metadata["error_code"] for d in mock.corpus}
    assert len(codes) == 12 * len(PROBLEMS)  # one per (service, problem), shared by 3 docs
    assert all(d.metadata["error_code"] in d.text for d in mock.corpus)


def test_qrels_are_graded_and_reference_real_docs(mock):
    ids = {d.id for d in mock.corpus}
    for rel in mock.qrels.values():
        assert set(rel) <= ids
        assert sorted(rel.values()) == [1, 1, 2]
        assert next(d for d, g in rel.items() if g == 2).endswith("/runbook")


def test_query_kinds_are_balanced(mock):
    counts = {k: list(mock.query_types.values()).count(k) for k in QUERY_KINDS}
    assert set(counts.values()) == {20}


def test_exact_queries_contain_the_code_paraphrases_do_not(mock):
    code_of = {d.id: d.metadata["error_code"] for d in mock.corpus}
    for qid, text in mock.queries.items():
        relevant_code = code_of[next(iter(mock.qrels[qid]))]
        if mock.query_types[qid].startswith("exact"):
            assert relevant_code in text
        else:
            assert relevant_code not in text


def test_scales_and_validates():
    small = generate_mock_dataset(n_services=3, n_problems=2, n_queries=100)
    assert len(small.corpus) == 3 * 2 * 3
    assert len(small.queries) == 6  # capped at the number of (service, problem) pairs
    with pytest.raises(ValueError):
        generate_mock_dataset(n_problems=99)
    with pytest.raises(ValueError):
        generate_mock_dataset(n_services=10_000)


def test_beir_roundtrip(mock, tmp_path):
    write_beir(mock, tmp_path)
    back = load_beir(tmp_path)
    assert [d.id for d in back.corpus] == [d.id for d in mock.corpus]
    assert back.corpus[0].metadata["error_code"] == mock.corpus[0].metadata["error_code"]
    assert back.qrels == mock.qrels
    assert json.loads((tmp_path / "query_types.json").read_text()) == mock.query_types
    # load_dataset picks the query kinds up as well
    assert load_dataset(str(tmp_path)).query_types == mock.query_types


def test_subset_keeps_query_types(mock):
    sub = mock.subset(10)
    assert set(sub.query_types) == set(sub.queries) and len(sub.queries) == 10


def test_lexical_and_dense_fail_in_different_places(mock):
    """The point of the dataset: BM25 owns exact identifiers, dense owns typos/paraphrase."""
    bm25 = BM25Retriever().index(mock.corpus)
    dense = DenseRetriever().index(mock.corpus)
    rb = evaluate(lambda q: bm25.search(q, 20), mock, ["ndcg@10"], "bm25")
    rd = evaluate(lambda q: dense.search(q, 20), mock, ["ndcg@10"], "dense")
    gb = group_metrics(rb, mock.query_types)
    gd = group_metrics(rd, mock.query_types)
    assert gb["exact"]["ndcg@10"] == pytest.approx(1.0)
    assert gb["exact"]["n"] == 20
    assert gd["typo"]["ndcg@10"] > gb["typo"]["ndcg@10"] + 0.2


def test_hybrid_is_at_least_as_good_as_its_weakest_leg_everywhere(mock):
    bm25 = BM25Retriever().index(mock.corpus)
    dense = DenseRetriever().index(mock.corpus)
    hybrid = HybridRetriever({"bm25": BM25Retriever(), "dense": DenseRetriever()})
    hybrid.index(mock.corpus)
    reports = {n: group_metrics(evaluate(lambda q, r=r: r.search(q, 20), mock, ["ndcg@10"], n),
                                mock.query_types)
               for n, r in {"bm25": bm25, "dense": dense, "hybrid": hybrid}.items()}
    for kind in QUERY_KINDS:
        legs = min(reports["bm25"][kind]["ndcg@10"], reports["dense"][kind]["ndcg@10"])
        assert reports["hybrid"][kind]["ndcg@10"] >= legs - 1e-9, kind
