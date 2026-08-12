"""Test doubles for the LLM layer: a deterministic fake client and a mock HTTP transport
that lets the real ``anthropic`` SDK run without a network."""

import json


class FakeLLM:
    """Deterministic stand-in for ClaudeClient: grades by keyword overlap."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system, prompt, schema):
        import re

        from hybrid_rag.llm import LLMError

        self.calls.append((system, prompt))
        if self.fail:
            raise LLMError("boom")
        query = re.search(r"<query>(.*?)</query>", prompt, re.S).group(1).lower().split()
        passages = re.findall(r'<passage id="(\d+)"[^>]*>\n(.*?)\n</passage>', prompt, re.S)
        overlap = {int(i): sum(w in text.lower() for w in query) for i, text in passages}
        if "judgments" in schema["properties"]:
            return {"judgments": [{"id": i, "relevance": min(10, 3 * s)}
                                  for i, s in overlap.items()]}
        return {"ranking": sorted(overlap, key=lambda i: -overlap[i])}


def message(content, stop_reason="end_turn"):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class Recorder:
    def __init__(self, responder):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.responder = responder

    def __call__(self, request):
        import httpx2

        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        return httpx2.Response(200, json=self.responder(self.requests[-1]))


def make_client(responder):
    """A ClaudeClient wired to the real SDK over a mock transport; returns (client, recorder)."""
    import anthropic
    import httpx2

    from hybrid_rag.llm import ClaudeClient

    rec = Recorder(responder)
    sdk = anthropic.Anthropic(
        api_key="sk-test", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(rec)),
    )
    return ClaudeClient(client=sdk), rec
