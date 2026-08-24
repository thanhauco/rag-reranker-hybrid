.PHONY: mock install install-all test test-all lint bench bench-models bench-scifact serve clean

install:            ## core + dev (offline: numpy, pyyaml)
	pip install -e ".[dev,api]"

install-all:        ## everything: models, faiss, Claude, API
	pip install -e ".[all,dev]"

test:               ## fast offline tests
	pytest -m "not models and not llm"

test-all:           ## includes tests that download model weights
	pytest

lint:
	ruff check src tests benchmarks examples

bench:              ## offline ablation on the bundled sample
	python benchmarks/run_benchmark.py

mock:               ## synthetic corpus -> ingest -> per-query-kind benchmark
	hybrid-rag mock --out data/mock
	hybrid-rag ingest data/mock/docs --query "$$(head -1 data/mock/queries.jsonl | python -c 'import json,sys; print(json.load(sys.stdin)["text"])')"
	python benchmarks/run_benchmark.py --dataset data/mock

bench-models:
	python benchmarks/run_benchmark.py --models

bench-scifact:      ## BEIR SciFact (5k docs, 300 queries) with neural models
	hybrid-rag download-beir scifact
	python benchmarks/run_benchmark.py --dataset data/beir/scifact --models

serve:
	hybrid-rag serve --config configs/hybrid_heuristic.yaml --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info **/__pycache__
