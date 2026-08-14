from hybrid_rag import BM25Retriever, ExtractiveGenerator, RAGPipeline
from hybrid_rag.rerankers import HeuristicReranker


def test_pipeline_search_and_answer(sample):
    pipe = RAGPipeline(BM25Retriever(), HeuristicReranker(), ExtractiveGenerator(),
                       retrieve_k=20, rerank_k=3).index(sample.corpus)
    resp = pipe.answer("what does exit code 137 mean in kubernetes")
    assert len(resp.results) == 3
    assert resp.candidates >= 3
    assert {"retrieve", "rerank", "generate", "total"} <= set(resp.timings_ms)
    assert resp.answer and resp.answer.citations
    assert resp.to_dict()["answer"]["citations"]
