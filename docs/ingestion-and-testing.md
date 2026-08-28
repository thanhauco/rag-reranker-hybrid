# Ingestion, mock data and testing

## Ingesting files

`hybrid_rag.ingest` turns a file or folder into `Document` objects; `ingest_path`
additionally (re)builds a pipeline's index from them.

| Format | One document per | Notes |
|---|---|---|
| `.md`, `.markdown`, `.txt` | file | optional YAML front matter (`---`) -> metadata; `id` and `title` keys honoured; otherwise title = first `# heading`, else the file name; id = relative path without extension |
| `.jsonl` | line | `{"id", "text", "title", ...}`; extra keys and a nested `"metadata"` object become metadata; missing id -> `<file>:<line>` |
| `.json` | list element | same record shape, top-level value must be a list |
| `.csv` | row | header must contain `text`; `id`, `title` optional; other columns become metadata |

```python
from hybrid_rag import RAGPipeline, SentenceChunker, BM25Retriever, ingest_path

pipe = RAGPipeline(BM25Retriever(), chunker=SentenceChunker(max_words=150))
stats = ingest_path(pipe, "docs/")
print(stats.summary())          # e.g. "214 files, 214 documents, 530 chunks, 61,204 words in 0.4s"
for err in stats.errors: ...    # unreadable files / malformed records, never silent
```

Behaviour worth knowing:

* **Nothing is half-ingested silently.** Unreadable files, invalid JSON lines, records
  without `text`, CSVs without a `text` column and files over 10 MB are skipped, logged
  and listed in `stats.errors`; empty documents are counted in `stats.skipped_empty`.
* **Duplicate ids raise** (`ValueError`), because two documents with one id would
  overwrite each other in every index and in evaluation.
* Hidden directories (`.git`, `.obsidian`, ...) and unsupported extensions are ignored.
* `ingest_path` raises if nothing usable was found, instead of producing an empty index.
* Chunking happens inside `pipeline.index`; chunks keep `parent_id`, and the evaluator
  maps chunk hits back to parent documents so chunked indexes score against
  document-level judgments.
* Re-ingesting rebuilds the index (replace semantics). To add to a dense index
  incrementally use `DenseRetriever.add`; the `/index` API endpoint also replaces.

CLI: `hybrid-rag ingest PATH [--config cfg.yaml] [--query "..."]...` prints the stats and
runs the queries; `--dataset PATH` on `search`/`answer`/`eval`/`serve` accepts a folder of
files too.

## Mock data

`hybrid_rag.mock` generates a deterministic synthetic knowledge base so the whole stack
(ingest -> index -> retrieve -> rerank -> evaluate -> serve) can be exercised without
downloading anything.

```python
from hybrid_rag import generate_mock_dataset
from hybrid_rag.mock import write_beir, write_markdown

ds = generate_mock_dataset(n_services=12, n_problems=10, n_queries=80, seed=0)
write_beir(ds, "data/mock")              # corpus.jsonl, queries.jsonl, qrels/test.tsv, query_types.json
write_markdown(ds, "data/mock/docs")     # <service>/<problem>/<kind>.md with front matter
```

**Corpus.** `n_services x n_problems x 3` documents (default 360). Each
(service, problem) has a runbook, postmortem and FAQ sharing one unique error code
(`DSK-1933`) and describing the problem with one of three phrasings.

**Relevance.** Graded: the runbook of the matching pair is 2, its postmortem and FAQ are
1. The hard negatives are the same problem on *other* services and other problems on the
same service.

**Queries.** Four kinds, equally represented, each built to favour one retrieval style:
`exact` (just the code), `exact_svc` (code + service), `paraphrase` (service + a rewording
that avoids the document vocabulary) and `typo` (misspelt service + content words).
`dataset.query_types` maps query ids to kinds and `group_metrics(report, dataset.query_types)`
breaks any evaluation down by kind - see `examples/06_ingest_mock_data.py` and
`python benchmarks/run_benchmark.py --dataset mock`.

Same arguments always give the same data (all randomness derives from `seed`).

## Test strategy

| Layer | Where | What it proves |
|---|---|---|
| Unit | `tests/test_{text,chunking,bm25,fusion,...}.py` | formulas (BM25 IDF floor, RRF, nDCG) against hand-computed values; edge cases (empty lists, ties, missing weights) |
| Property of the data | `tests/test_mock.py` | generator is deterministic; qrels reference real docs and are graded `[2,1,1]`; BM25 owns exact queries, dense owns typos; **hybrid is never worse than its weaker leg on any query kind** |
| Ingestion | `tests/test_ingest.py` | every format, front matter, error accounting, duplicate ids, size cap, hidden files; **Markdown round-trip preserves ids and metadata so qrels still line up** |
| Integration | `tests/test_ingest.py`, `tests/test_pipeline.py` | CLI `mock -> ingest -> eval`, `/index` + `/search` through FastAPI, YAML configs build and run |
| Claude | `tests/test_claude_sdk.py`, `tests/test_claude_generation.py` | the real `anthropic` SDK against a mock HTTP transport: request bodies (model, fallbacks beta, structured-output schema, citation documents) and response parsing, with no network and no spend |
| Models | `-m models` | real cross-encoder end to end (downloads ~90 MB) |

```bash
make test        # everything offline
make test-all    # + model-backed tests
make mock        # generate data/mock, ingest it, run the per-kind benchmark
```
