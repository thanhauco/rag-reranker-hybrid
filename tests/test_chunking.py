import pytest

from hybrid_rag import Document, FixedSizeChunker, RecursiveChunker, SentenceChunker


def test_chunkers_keep_parent_pointer():
    doc = Document("p", " ".join(f"Sentence number {i} is here." for i in range(50)),
                   {"title": "T"})
    for chunker in (FixedSizeChunker(40, 10), SentenceChunker(40, 1), RecursiveChunker(40)):
        chunks = chunker.split([doc])
        assert len(chunks) > 1
        assert all(c.metadata["parent_id"] == "p" and c.metadata["title"] == "T" for c in chunks)
        assert all(len(c.text.split()) <= 45 for c in chunks)


def test_fixed_chunker_validates_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(10, 10)
