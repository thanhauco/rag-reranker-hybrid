import json

from hybrid_rag.cli import main


def test_cli_mock_then_ingest_then_eval(tmp_path, capsys):
    out = tmp_path / "m"
    main(["mock", "--out", str(out), "--services", "4", "--problems", "3", "--queries", "12"])
    assert "36 files" in capsys.readouterr().out
    assert (out / "corpus.jsonl").exists() and len(list((out / "docs").rglob("*.md"))) == 36

    code = json.loads((out / "corpus.jsonl").read_text().splitlines()[0])["metadata"]["error_code"]
    main(["ingest", str(out / "docs"), "--query", code, "-k", "1"])
    text = capsys.readouterr().out
    assert "ingested 36 files, 36 documents" in text and f"Q: {code}" in text

    main(["eval", "--dataset", str(out), "--metrics", "ndcg@10"])
    text = capsys.readouterr().out
    assert "By query type:" in text and "exact" in text and "paraphrase" in text



def test_cli_search_json(capsys):
    main(["search", "allkeys-lru eviction", "-k", "2", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["id"] == "d09" and len(out["results"]) == 2
    assert {"retrieve", "rerank", "total"} <= set(out["timings_ms"])


def test_cli_answer_prints_text_and_citations(capsys):
    main(["answer", "what does allkeys-lru do", "-k", "2"])
    out = capsys.readouterr().out
    assert "allkeys-lru" in out and "[d09]" in out


def test_cli_eval_default_pipeline_on_sample(capsys):
    main(["eval", "--metrics", "ndcg@10", "hit_rate@1"])
    out = capsys.readouterr().out
    assert "sample-kb" in out and "| default |" in out
