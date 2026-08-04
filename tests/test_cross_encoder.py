import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hybrid_rag.rerankers import HTTPReranker
from tests.helpers import CANDIDATES, QUERY


@pytest.mark.models
def test_cross_encoder_real_model():
    pytest.importorskip("sentence_transformers")
    from hybrid_rag.rerankers import CrossEncoderReranker

    r = CrossEncoderReranker("minilm")
    out = r.rerank(QUERY, CANDIDATES)
    assert out[0].id == "d2"
    assert all(0.0 <= x.score <= 1.0 for x in out)



def test_http_reranker_batches_and_maps_indices():
    calls = []

    class Handler(BaseHTTPRequestHandler):  # a minimal TEI /rerank server
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            query = set(body["query"].split())
            items = [{"index": i, "score": float(len(query & set(t.split())))}
                     for i, t in enumerate(body["texts"])]
            items.sort(key=lambda x: -x["score"])  # TEI returns results sorted by score
            payload = json.dumps(items).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reranker = HTTPReranker(f"http://127.0.0.1:{server.server_port}", batch_size=2)
        out = reranker.rerank(QUERY, CANDIDATES)
    finally:
        server.shutdown()
        server.server_close()
    assert out[0].id == "d2"
    assert len(calls) == 2 and len(calls[0]["texts"]) == 2  # 3 docs, batch_size 2
    assert all(r.scores["http"] == r.score for r in out)
