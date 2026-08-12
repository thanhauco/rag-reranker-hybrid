"""Drive the real ``anthropic`` SDK against a mock HTTP transport: verifies the request
bodies we send (model, fallbacks beta, structured outputs) and that real SDK responses
parse - with no network and no API spend."""

import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from types import SimpleNamespace  # noqa: E402

from hybrid_rag import BM25Retriever  # noqa: E402
from hybrid_rag.llm import FALLBACK_BETA, ClaudeClient, LLMError  # noqa: E402
from hybrid_rag.rerankers import LLMListwiseReranker, LLMPointwiseReranker  # noqa: E402
from tests.fakes import make_client, message  # noqa: E402


def test_listwise_request_shape_and_parse(sample):
    def respond(body):
        return message([{"type": "text", "text": json.dumps({"ranking": [3, 1, 2]})}])

    client, rec = make_client(respond)
    bm25 = BM25Retriever().index(sample.corpus)
    cands = bm25.search("kubernetes pod restart", 3)
    out = LLMListwiseReranker(client, window_size=20).rerank("kubernetes pod restart", cands)
    assert [r.id for r in out] == [cands[2].id, cands[0].id, cands[1].id]

    body, headers = rec.requests[0], rec.headers[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in headers["anthropic-beta"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body  # always-on adaptive thinking on Opus 5.5: omit the param
    assert "<passage id=\"1\"" in body["messages"][0]["content"]


def test_pointwise_refusal_degrades_to_first_stage_order(sample):
    client, _ = make_client(lambda body: message([], stop_reason="refusal"))
    cands = BM25Retriever().index(sample.corpus).search("redis eviction", 4)
    out = LLMPointwiseReranker(client).rerank("redis eviction", cands)
    assert [r.id for r in out] == [c.id for c in cands]


def test_api_errors_map_to_llm_error():
    rec_client = anthropic.Anthropic(
        api_key="sk-test", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(
            lambda req: httpx2.Response(529, json={"type": "error", "error": {
                "type": "overloaded_error", "message": "Overloaded"}}))),
    )
    with pytest.raises(LLMError, match="529"):
        ClaudeClient(client=rec_client).complete_json("s", "p", {"type": "object"})



def test_claude_client_refusal_raises():
    from hybrid_rag.llm import ClaudeClient, LLMError

    resp = SimpleNamespace(stop_reason="refusal",
                           stop_details=SimpleNamespace(category="cyber"))
    with pytest.raises(LLMError, match="cyber"):
        ClaudeClient.check_stop(resp)
