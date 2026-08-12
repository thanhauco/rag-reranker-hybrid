"""Thin Claude client used by the LLM reranker and the grounded generator.

Everything LLM-facing goes through the small :class:`LLMClient` protocol so tests (and
other backends) can swap in a fake without touching reranker logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side refusal fallback: on a safety decline, the API re-runs the request on
# Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Any failure to obtain a usable LLM response (API error, refusal, truncation)."""


class LLMClient(Protocol):
    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object that validates against ``schema``."""
        ...


class ClaudeClient:
    """Claude via the official ``anthropic`` SDK.

    Credentials resolve from the environment (``ANTHROPIC_API_KEY``, or an
    ``ant auth login`` profile). Thinking is adaptive and always on for Claude Opus 5.5;
    ``effort`` is the cost/quality knob. Reranking is a judgement-per-passage task, so
    ``low`` is a sensible default - raise it if your eval shows headroom.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 16000,
        use_fallbacks: bool = True,
        client: Any = None,
    ):
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'hybrid-rag[llm]' to use Claude") from e
            client = anthropic.Anthropic()
        self.client = client
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.use_fallbacks = use_fallbacks

    def create(self, **kwargs: Any) -> Any:
        """``messages.create`` with refusal fallbacks enabled (when configured) and SDK
        errors mapped to :class:`LLMError`."""
        import anthropic

        kwargs.setdefault("model", self.model)
        kwargs.setdefault("max_tokens", self.max_tokens)
        try:
            if self.use_fallbacks:
                return self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            return self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited (after SDK retries): {e.message}") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"bad request: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e

    @staticmethod
    def check_stop(response: Any) -> None:
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMError(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self.create(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        self.check_stop(response)
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in structured response")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid JSON from model: {e}") from e
