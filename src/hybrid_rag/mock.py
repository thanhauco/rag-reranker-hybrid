"""Deterministic synthetic corpora with labelled queries - for tests, demos and benchmarks.

The corpus is a fictional engineering knowledge base. Every (service, problem) pair gets
three documents - a runbook, a postmortem and an FAQ - that share a unique error code and
describe the problem in one of several *different phrasings*. Queries come in four kinds,
chosen so that lexical and dense retrieval fail in different places:

============  ===========================================  =========================
kind          example                                      who should win
============  ===========================================  =========================
``exact``     ``BIL-4821``                                 BM25 (rare identifier)
``exact_svc`` ``BIL-4821 billing-api``                     BM25
``paraphrase`` ``why does billing-api stop answering``     dense (no shared symptom words)
``typo``      ``bilng-api requests time out``              dense / char n-grams
============  ===========================================  =========================

Relevance is graded: the runbook for the matching (service, problem) is 2, its postmortem
and FAQ are 1; same-problem documents of *other* services are the hard negatives.

Everything derives from ``seed``, so the same arguments always produce the same data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .evaluation.datasets import RetrievalDataset, load_beir, load_sample
from .types import Document

_PREFIXES = ["billing", "auth", "search", "notification", "order", "inventory", "media",
             "report", "payment", "profile", "audit", "recommendation", "shipping",
             "pricing", "catalog", "session"]
_SUFFIXES = ["api", "service", "worker", "gateway", "indexer", "scheduler", "cache", "sync"]


@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    code: str
    symptoms: tuple[str, ...]          # alternative phrasings used inside documents
    paraphrases: tuple[str, ...]       # query phrasings that avoid the symptom vocabulary
    typo_terms: tuple[str, ...]        # content words kept (not misspelled) in typo queries
    remedy: str
    cause: str


PROBLEMS: tuple[Problem, ...] = (
    Problem("timeout", "request timeouts", "TMO",
            ("requests time out after thirty seconds", "calls fail with deadline exceeded",
             "responses do not arrive within the allowed window"),
            ("why does {svc} stop answering", "{svc} is too slow and callers give up"),
            ("requests", "time", "out"),
            "Add a circuit breaker, retry with jittered exponential backoff and raise the "
            "upstream deadline only after ruling out slow queries.",
            "a downstream dependency saturated its worker threads"),
    Problem("memory_leak", "memory leaks", "MEM",
            ("resident memory grows steadily until the pod is OOMKilled",
             "heap usage climbs and the kernel out-of-memory killer ends the process",
             "the container keeps getting restarted with exit code 137"),
            ("{svc} keeps getting killed after running for a few hours",
             "{svc} uses more and more RAM over time"),
            ("memory", "restarted"),
            "Capture a heap profile before the restart, bound every in-process cache and "
            "set the memory request close to steady-state usage.",
            "an unbounded in-process cache was never evicted"),
    Problem("conn_exhaustion", "connection pool exhaustion", "CON",
            ("the database connection pool is exhausted",
             "clients receive too many connections errors",
             "new queries wait forever for a free connection"),
            ("{svc} cannot reach its database even though the database is healthy",
             "database access from {svc} hangs under load"),
            ("database", "connections"),
            "Put a pooler such as PgBouncer in front of the database and audit code paths "
            "that never return connections.",
            "connections leaked on an error path"),
    Problem("cert_expiry", "certificate expiry", "TLS",
            ("the TLS certificate has expired", "handshakes fail with x509 certificate is "
             "not valid", "clients reject the server certificate"),
            ("secure calls to {svc} suddenly stopped working",
             "{svc} is refusing encrypted traffic since this morning"),
            ("certificate", "handshake"),
            "Renew the certificate, automate rotation with cert-manager and alert thirty "
            "days before expiry.",
            "the renewal job had been failing silently"),
    Problem("disk_full", "disk exhaustion", "DSK",
            ("the volume has no space left on device", "writes fail with ENOSPC",
             "the data disk reached one hundred percent utilisation"),
            ("{svc} cannot save anything anymore", "{svc} stopped persisting new data"),
            ("disk", "writes"),
            "Free space by rotating logs, expand the volume and add an alert at eighty "
            "percent utilisation.",
            "debug logging was left enabled after an incident"),
    Problem("rate_limit", "rate limiting", "RTE",
            ("callers receive HTTP 429 too many requests",
             "the quota for the client key has been exceeded",
             "traffic is throttled once the burst allowance is spent"),
            ("{svc} rejects clients that send a lot of calls",
             "partners complain that {svc} turns them away during peaks"),
            ("limit", "quota"),
            "Honour the Retry-After header, use a token bucket on the client and request a "
            "higher quota for batch jobs.",
            "a batch job ignored the published limits"),
    Problem("queue_lag", "queue backlog", "QUE",
            ("consumer lag keeps growing on the main topic",
             "the message backlog is draining slower than it fills",
             "events are processed minutes after they were published"),
            ("{svc} is falling behind on its incoming work",
             "updates from {svc} show up very late"),
            ("consumer", "backlog"),
            "Scale consumers up to the partition count, fix slow handlers and move poison "
            "messages to a dead-letter queue.",
            "one slow handler blocked a partition"),
    Problem("deadlock", "database deadlocks", "DLK",
            ("transactions abort with deadlock detected",
             "two writers lock rows in opposite order and both stall",
             "lock wait timeout exceeded appears under concurrent updates"),
            ("{svc} randomly fails writes when many users edit the same records",
             "concurrent edits make {svc} error out"),
            ("transactions", "lock"),
            "Acquire locks in a consistent order, keep transactions short and retry on "
            "serialization failures.",
            "two code paths updated the same rows in different order"),
    Problem("clock_skew", "clock skew", "CLK",
            ("server clocks drift apart by several minutes",
             "tokens are rejected as issued in the future",
             "NTP synchronisation has stopped on some nodes"),
            ("{svc} rejects valid logins on some machines only",
             "{svc} behaves differently depending on which node answers"),
            ("clock", "tokens"),
            "Restart chronyd, alert on offset above one second and allow small leeway when "
            "validating token timestamps.",
            "a node lost access to its time source"),
    Problem("bad_deploy", "faulty deployments", "DEP",
            ("error rate spiked right after the latest rollout",
             "the new release returns HTTP 500 for a subset of routes",
             "the canary is failing its health checks"),
            ("{svc} broke after we shipped yesterday", "things went wrong in {svc} after the "
             "update went out"),
            ("release", "rollout"),
            "Roll back to the previous version, keep canaries at five percent until error "
            "rates are flat and add a regression test.",
            "a configuration value was missing in the new release"),
)

KINDS = ("runbook", "postmortem", "faq")
QUERY_KINDS = ("exact", "exact_svc", "paraphrase", "typo")
_RELEVANCE = {"runbook": 2, "postmortem": 1, "faq": 1}


def _service_names(n: int, rng: random.Random) -> list[str]:
    names = [f"{p}-{s}" for p in _PREFIXES for s in _SUFFIXES]
    if n > len(names):
        raise ValueError(f"at most {len(names)} services supported, got {n}")
    rng.shuffle(names)
    return sorted(names[:n])


def _typo(word: str, rng: random.Random) -> str:
    """One realistic slip: transpose two letters, drop one, or double one."""
    letters = [i for i, c in enumerate(word) if c.isalpha()]
    if len(letters) < 4:
        return word
    i = rng.choice(letters[1:-1])
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap" and word[i + 1].isalpha():
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + word[i + 1:]


def _body(kind: str, svc: str, prob: Problem, code: str, version: str,
          rng: random.Random) -> tuple[str, str]:
    symptom = rng.choice(prob.symptoms)
    if kind == "runbook":
        title = f"{svc}: {prob.title} runbook"
        text = (f"Error {code} on {svc} (seen since {version}). Symptom: {symptom}. "
                f"Likely cause: {prob.cause}. Mitigation: {prob.remedy} "
                f"Escalate to the {svc.split('-')[0]} on-call if {code} persists after the fix.")
    elif kind == "postmortem":
        title = f"Postmortem: {prob.title} in {svc}"
        text = (f"Summary: during the incident {code} the {svc} team saw that {symptom}. "
                f"Root cause: {prob.cause}. Impact lasted forty minutes and affected "
                f"dependent services. Follow-ups: {prob.remedy}")
    else:
        title = f"FAQ: {prob.title} in {svc}"
        text = (f"Q: What does {code} mean for {svc}? A: It indicates that {symptom}. "
                f"Q: How do I fix it? A: {prob.remedy}")
    return title, text


def generate_mock_dataset(n_services: int = 12, n_problems: int = len(PROBLEMS),
                          n_queries: int = 80, seed: int = 0) -> RetrievalDataset:
    """Build ``n_services * n_problems * 3`` documents and up to ``n_queries`` queries.

    Defaults give 360 documents. Queries cycle through :data:`QUERY_KINDS` over a shuffled
    sample of (service, problem) pairs, so every kind is represented equally.
    """
    if not 1 <= n_problems <= len(PROBLEMS):
        raise ValueError(f"n_problems must be in [1, {len(PROBLEMS)}]")
    rng = random.Random(seed)
    services = _service_names(n_services, rng)
    problems = PROBLEMS[:n_problems]

    corpus: list[Document] = []
    pair_docs: dict[tuple[str, str], dict[str, str]] = {}
    pair_code: dict[tuple[str, str], str] = {}
    used_codes: set[str] = set()
    for svc in services:
        for prob in problems:
            while True:
                code = f"{prob.code}-{rng.randint(1000, 9999)}"
                if code not in used_codes:
                    used_codes.add(code)
                    break
            pair_code[(svc, prob.key)] = code
            version = f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}"
            for kind in KINDS:
                title, text = _body(kind, svc, prob, code, version, rng)
                doc_id = f"{svc}/{prob.key}/{kind}"
                corpus.append(Document(doc_id, text, {
                    "title": title, "service": svc, "problem": prob.key, "kind": kind,
                    "error_code": code}))
                pair_docs.setdefault((svc, prob.key), {})[kind] = doc_id

    pairs = [(s, p) for s in services for p in problems]
    rng.shuffle(pairs)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    qtypes: dict[str, str] = {}
    for i, (svc, prob) in enumerate(pairs[:n_queries]):
        kind = QUERY_KINDS[i % len(QUERY_KINDS)]
        code = pair_code[(svc, prob.key)]
        if kind == "exact":
            text = code
        elif kind == "exact_svc":
            text = f"{code} {svc}"
        elif kind == "paraphrase":
            text = rng.choice(prob.paraphrases).format(svc=svc)
        else:
            text = f"{_typo(svc, rng)} {' '.join(prob.typo_terms)}"
        qid = f"mq{i:04d}"
        queries[qid] = text
        qtypes[qid] = kind
        qrels[qid] = {doc: _RELEVANCE[k] for k, doc in pair_docs[(svc, prob.key)].items()}
    return RetrievalDataset(f"mock-{n_services}x{n_problems}", corpus, queries, qrels, qtypes)


# --------------------------------------------------------------------------- writing
def write_beir(ds: RetrievalDataset, out: str | Path) -> Path:
    """Write ``corpus.jsonl``, ``queries.jsonl``, ``qrels/test.tsv`` (+ ``query_types.json``)."""
    out = Path(out)
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for d in ds.corpus:
            meta = {k: v for k, v in d.metadata.items() if k != "title"}
            f.write(json.dumps({"_id": d.id, "title": d.title, "text": d.text,
                                "metadata": meta}) + "\n")
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q, t in ds.queries.items():
            f.write(json.dumps({"_id": q, "text": t}) + "\n")
    with (out / "qrels" / "test.tsv").open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, rel in ds.qrels.items():
            for d, g in rel.items():
                f.write(f"{q}\t{d}\t{g}\n")
    (out / "query_types.json").write_text(json.dumps(ds.query_types, indent=2))
    return out


def write_markdown(ds: RetrievalDataset, out: str | Path) -> Path:
    """One Markdown file per document with YAML front matter - the shape real wikis and
    docs folders have, and what :func:`hybrid_rag.ingest.load_documents` reads."""
    out = Path(out)
    for d in ds.corpus:
        path = out / f"{d.id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        front = {"id": d.id, **{k: v for k, v in d.metadata.items() if k != "title"}}
        path.write_text(f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n"
                        f"# {d.title}\n\n{d.text}\n", encoding="utf-8")
    return out


def load_dataset(spec: str) -> RetrievalDataset:
    """Resolve a dataset spec: ``"sample"``, ``"mock"``, a BEIR directory, or any folder /
    file of raw documents (ingested without queries, so it can be searched but not scored).
    """
    if spec == "sample":
        return load_sample()
    if spec == "mock":
        return generate_mock_dataset()
    path = Path(spec)
    if (path / "corpus.jsonl").exists() and (path / "queries.jsonl").exists():
        ds = load_beir(path)
        types = path / "query_types.json"
        if types.exists():
            ds.query_types = json.loads(types.read_text())
        return ds
    from .ingest import load_documents

    return RetrievalDataset(path.name or "documents", load_documents(path), {}, {})
