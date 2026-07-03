"""Lightweight, dependency-free text processing for lexical retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

# Keeps identifiers such as ``ERR_CONN_RESET``, ``gpt-4o``, ``v1.2`` and ``429`` intact
# as single tokens: these exact-match terms are where BM25 beats dense retrieval.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    yours yourself yourselves
    """.split()
)

_SUFFIXES = ("ational", "ization", "fulness", "iveness", "ations", "ation", "ments", "ment",
             "ness", "ings", "ing", "ies", "ied", "ers", "est", "ed", "es", "ly", "er", "s")


def light_stem(token: str) -> str:
    """A conservative suffix stripper (much lighter than Porter).

    Good enough to conflate ``index/indexes/indexing`` while leaving short tokens,
    numbers and identifiers untouched.
    """
    if len(token) <= 4 or not token.isalpha():
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            stem = token[: -len(suf)]
            if suf in ("ies", "ied"):
                stem += "y"
            return stem
    return token


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> list[str]:
    if lowercase:
        text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens


Tokenizer = Callable[[str], list[str]]


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Iterable[str]:
    """Character n-grams over word boundaries (``#word#``) - robust to typos/morphology."""
    for word in _TOKEN_RE.findall(text.lower()):
        padded = f"#{word}#"
        for n in range(n_min, n_max + 1):
            if len(padded) < n:
                break
            for i in range(len(padded) - n + 1):
                yield padded[i : i + n]
