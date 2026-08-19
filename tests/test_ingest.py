import json

import pytest

from hybrid_rag import (
    BM25Retriever,
    RAGPipeline,
    SentenceChunker,
    ingest_path,
    load_documents,
)
from hybrid_rag.ingest import IngestStats, split_front_matter
from hybrid_rag.mock import generate_mock_dataset, write_markdown
from hybrid_rag.rerankers import HeuristicReranker


def test_front_matter():
    meta, body = split_front_matter("---\nid: x\ntags: [a, b]\n---\nhello")
    assert meta == {"id": "x", "tags": ["a", "b"]} and body == "hello"
    assert split_front_matter("no front matter") == ({}, "no front matter")
    with pytest.raises(ValueError):
        split_front_matter("---\n- a list\n---\nbody")


def test_markdown_title_from_heading_and_filename(tmp_path):
    (tmp_path / "a.md").write_text("# Pod restarts\n\nBody about restarts.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "my_notes-file.txt").write_text("plain text only")
    docs = {d.id: d for d in load_documents(tmp_path)}
    assert docs["a"].title == "Pod restarts" and docs["a"].text == "Body about restarts."
    assert docs["sub/my_notes-file"].title == "my notes file"
    assert docs["a"].metadata["source"] == "a.md"


def test_front_matter_overrides_id_and_title_and_dedupes_heading(tmp_path):
    (tmp_path / "x.md").write_text("---\nid: custom-1\ntitle: Real Title\nteam: infra\n---\n"
                                   "# Real Title\n\nText.")
    (doc,) = load_documents(tmp_path)
    assert (doc.id, doc.title, doc.text) == ("custom-1", "Real Title", "Text.")
    assert doc.metadata["team"] == "infra"


def test_jsonl_json_csv(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"id": "j1", "text": "one", "title": "T", "lang": "en"}) + "\n\n"
        + json.dumps({"text": "two", "metadata": {"k": 1}}) + "\n")
    (tmp_path / "b.json").write_text(json.dumps([{"_id": "k1", "text": "three"}]))
    (tmp_path / "c.csv").write_text("id,title,text,owner\nc1,CT,four,ann\n,,five,\n")
    docs = {d.id: d for d in load_documents(tmp_path)}
    assert docs["j1"].metadata["lang"] == "en" and docs["j1"].title == "T"
    assert docs["a:3"].metadata["k"] == 1  # id falls back to <file>:<line>
    assert docs["k1"].text == "three"
    assert docs["c1"].metadata["owner"] == "ann" and docs["c:2"].text == "five"
    assert len(docs) == 5


def test_bad_records_are_counted_not_fatal(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"text": "ok"}\nnot json\n{"no_text": 1}\n')
    (tmp_path / "bad.json").write_text("{not valid")
    (tmp_path / "nocol.csv").write_text("a,b\n1,2\n")
    (tmp_path / "empty.md").write_text("---\nid: e\n---\n\n")
    stats = IngestStats()
    docs = load_documents(tmp_path, stats)
    assert [d.text for d in docs] == ["ok"]
    assert stats.skipped_empty == 1
    assert len(stats.errors) == 4
    assert any("no 'text'" in e for e in stats.errors)


def test_duplicate_ids_raise(tmp_path):
    (tmp_path / "a.md").write_text("---\nid: same\n---\nfirst")
    (tmp_path / "b.md").write_text("---\nid: same\n---\nsecond")
    with pytest.raises(ValueError, match="duplicate document id 'same'"):
        load_documents(tmp_path)


def test_hidden_and_unsupported_files_ignored(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.md").write_text("secret")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "ok.md").write_text("fine")
    assert [d.id for d in load_documents(tmp_path)] == ["ok"]


def test_missing_path_and_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
    with pytest.raises(ValueError, match="no documents"):
        ingest_path(RAGPipeline(BM25Retriever()), tmp_path)


def test_oversized_file_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("hybrid_rag.ingest.MAX_BYTES", 10)
    (tmp_path / "big.md").write_text("x" * 100)
    stats = IngestStats()
    assert load_documents(tmp_path, stats) == [] and "larger than" in stats.errors[0]


def test_ingest_mock_markdown_end_to_end(tmp_path):
    ds = generate_mock_dataset(n_services=4, n_problems=3, n_queries=12)
    write_markdown(ds, tmp_path)
    pipe = RAGPipeline(BM25Retriever(), HeuristicReranker(), rerank_k=3)
    stats = ingest_path(pipe, tmp_path)
    assert stats.documents == stats.chunks == len(ds.corpus) == 36
    assert stats.files == 36 and not stats.errors and "36 documents" in stats.summary()
    # ids and metadata survive the file round trip, so qrels still line up
    ingested = {d.id: d for d in pipe.retriever.documents}
    assert set(ingested) == {d.id for d in ds.corpus}
    first = ds.corpus[0]
    assert ingested[first.id].metadata["error_code"] == first.metadata["error_code"]
    assert ingested[first.id].title == first.title
    # a search by error code finds the right service/problem
    top = pipe.search(first.metadata["error_code"]).results[0]
    assert top.document.metadata["error_code"] == first.metadata["error_code"]


def test_ingest_with_chunker_reports_chunks(tmp_path):
    long_text = " ".join(f"Sentence number {i} talks about caching." for i in range(120))
    (tmp_path / "long.md").write_text(f"# Long\n\n{long_text}")
    pipe = RAGPipeline(BM25Retriever(), chunker=SentenceChunker(max_words=40))
    stats = ingest_path(pipe, tmp_path)
    assert stats.documents == 1 and stats.chunks > 3
    assert pipe.retriever.documents[0].metadata["parent_id"] == "long"
