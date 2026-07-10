"""BM25 over an inverted index, vectorised with numpy.

Variants (Trotman et al., 2014, "Improvements to BM25 and Language Models Examined"):

* ``okapi`` - classic Robertson/Sparck-Jones IDF, floored at ``epsilon * mean_idf``
* ``plus``  - BM25+ adds ``delta`` so long documents are not over-penalised
* ``l``     - BM25L shifts the normalised tf by ``delta``
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..text import Tokenizer, tokenize
from ..types import Document, SearchResult
from .base import Retriever


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        variant: str = "okapi",
        delta: float = 1.0,
        epsilon: float = 0.25,
        tokenizer: Tokenizer = tokenize,
        index_fields: tuple[str, ...] = ("title",),
    ):
        super().__init__()
        if variant not in ("okapi", "plus", "l"):
            raise ValueError("variant must be 'okapi', 'plus' or 'l'")
        self.k1, self.b, self.variant, self.delta, self.epsilon = k1, b, variant, delta, epsilon
        self.tokenizer = tokenizer
        self.index_fields = index_fields
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(0)
        self._avgdl = 0.0

    def _doc_text(self, doc: Document) -> str:
        extra = " ".join(str(doc.metadata.get(f, "")) for f in self.index_fields)
        return f"{extra} {doc.text}" if extra.strip() else doc.text

    def index(self, documents: Iterable[Document]) -> BM25Retriever:
        self.documents = list(documents)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, doc in enumerate(self.documents):
            toks = self.tokenizer(self._doc_text(doc))
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self._doc_len = np.asarray(lengths, dtype=np.float32)
        self._avgdl = float(self._doc_len.mean()) if len(lengths) else 0.0
        self._postings = {
            t: (np.fromiter((d for d, _ in p), np.int32, len(p)),
                np.fromiter((f for _, f in p), np.float32, len(p)))
            for t, p in postings.items()
        }
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        n = len(self.documents)
        idf = {}
        for term, (docs, _) in self._postings.items():
            df = len(docs)
            if self.variant == "okapi":
                idf[term] = math.log((n - df + 0.5) / (df + 0.5))
            else:  # BM25+/BM25L use the always-positive form
                idf[term] = math.log((n + 1) / (df + 0.5))
        if self.variant == "okapi" and idf:
            # Terms in > half the corpus get negative IDF; floor them at a fraction of the
            # mean positive IDF (rank_bm25 uses the overall mean, which can itself be
            # negative on small corpora and flip scores).
            positive = [v for v in idf.values() if v > 0]
            floor = self.epsilon * (sum(positive) / len(positive) if positive else 1.0)
            idf = {t: (v if v > 0 else floor) for t, v in idf.items()}
        self._idf = idf

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not self.documents:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self._doc_len / max(self._avgdl, 1e-9))
        for term, qtf in Counter(self.tokenizer(query)).items():
            if term not in self._postings:
                continue
            docs, tf = self._postings[term]
            idf = self._idf[term]
            n = norm[docs]
            if self.variant == "okapi":
                s = tf * (self.k1 + 1) / (tf + n)
            elif self.variant == "plus":
                s = tf * (self.k1 + 1) / (tf + n) + self.delta
            else:  # BM25L
                ctd = tf / (1 - self.b + self.b * self._doc_len[docs] / self._avgdl)
                s = (self.k1 + 1) * (ctd + self.delta) / (self.k1 + ctd + self.delta)
            scores[docs] += idf * s * qtf
        return scores

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        scores = self.get_scores(query)
        results = self._to_results(scores, k)
        return [r for r in results if r.score > 0]  # no term overlap -> not a match

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        state = {k: v for k, v in self.__dict__.items() if k != "tokenizer"}
        Path(path).write_bytes(pickle.dumps(state))

    @classmethod
    def load(cls, path: str | Path, tokenizer: Tokenizer = tokenize) -> BM25Retriever:
        """Load a pickled index. Only load files you created - pickle is not safe for
        untrusted input."""
        obj = cls.__new__(cls)
        obj.__dict__.update(pickle.loads(Path(path).read_bytes()))
        obj.tokenizer = tokenizer
        return obj
