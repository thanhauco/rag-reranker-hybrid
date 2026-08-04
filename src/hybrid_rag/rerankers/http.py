"""Remote reranker over HTTP, compatible with Hugging Face Text Embeddings Inference
(TEI) ``POST /rerank``. Serve any cross-encoder on a GPU box and call it from here:

    docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \\
        --model-id BAAI/bge-reranker-base
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence

from ..types import Document
from .base import Reranker


class HTTPReranker(Reranker):
    name = "http"

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 30.0,
                 batch_size: int = 32, headers: dict[str, str] | None = None,
                 top_k: int | None = None, min_score: float | None = None):
        super().__init__(top_k, min_score)
        self.url = url.rstrip("/") + "/rerank"
        self.timeout, self.batch_size = timeout, batch_size
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def _post(self, payload: dict) -> list[dict]:
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        scores = [0.0] * len(documents)
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            items = self._post({"query": query, "texts": [d.text for d in batch],
                                "raw_scores": False, "truncate": True})
            for item in items:
                scores[start + int(item["index"])] = float(item["score"])
        return scores
