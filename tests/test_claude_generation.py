import pytest

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("httpx2")

from types import SimpleNamespace  # noqa: E402

from hybrid_rag import BM25Retriever, ClaudeGenerator, RAGPipeline  # noqa: E402
from tests.fakes import make_client, message  # noqa: E402


def _fake_claude_response(contexts_titles):
    cit = SimpleNamespace(document_index=1, cited_text="exit code 137", document_title="t")
    blocks = [SimpleNamespace(type="text", text="It was OOM killed.", citations=[cit])]
    return SimpleNamespace(content=blocks, stop_reason="end_turn", stop_details=None,
                           model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_claude_generator_builds_citation_request_and_parses(sample):
    from hybrid_rag.llm import ClaudeClient

    captured = {}

    class FakeSDK:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return _fake_claude_response(None)

    pipe = RAGPipeline(BM25Retriever(), rerank_k=2).index(sample.corpus)
    results = pipe.search("exit code 137").results
    gen = ClaudeGenerator(ClaudeClient(client=FakeSDK()))
    ans = gen.generate("exit code 137?", results)
    assert ans.text == "It was OOM killed."
    assert ans.citations[0].doc_id == results[1].id
    docs = [b for b in captured["messages"][0]["content"] if b["type"] == "document"]
    assert len(docs) == 2 and all(d["citations"] == {"enabled": True} for d in docs)
    assert captured["model"] == "claude-opus-5-5"
    assert captured["fallbacks"] == "default"



def test_generator_sends_citation_documents_and_parses_citations(sample):
    def respond(body):
        return message([
            {"type": "text", "text": "Use allkeys-lru for caches.", "citations": [{
                "type": "char_location", "cited_text": "allkeys-lru evicts the least recently",
                "document_index": 0, "document_title": "Redis eviction policies",
                "start_char_index": 0, "end_char_index": 10,
            }]},
        ])

    client, rec = make_client(respond)
    pipe = RAGPipeline(BM25Retriever(), generator=ClaudeGenerator(client), rerank_k=2)
    resp = pipe.index(sample.corpus).answer("which redis eviction policy for a cache")
    assert resp.answer.text == "Use allkeys-lru for caches."
    assert resp.answer.citations[0].doc_id == resp.results[0].id
    content = rec.requests[0]["messages"][0]["content"]
    docs = [b for b in content if b["type"] == "document"]
    assert len(docs) == 2 and docs[0]["citations"] == {"enabled": True}
    assert docs[0]["source"]["type"] == "text"
    assert content[-1] == {"type": "text", "text": "which redis eviction policy for a cache"}
