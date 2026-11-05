from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever


def test_bm25_prefers_exact_keyword_match():
    chunks = [
        DocumentChunk(
            text="Python is commonly used for machine learning.",
            source="python.txt",
            chunk_id="python:1",
        ),
        DocumentChunk(
            text="PostgreSQL supports B-tree and GIN indexes.",
            source="database.txt",
            chunk_id="database:1",
        ),
        DocumentChunk(
            text="Neural networks learn representations from data.",
            source="deep-learning.txt",
            chunk_id="dl:1",
        ),
    ]

    retriever = BM25Retriever()
    retriever.add(chunks)

    results = retriever.search(
        "PostgreSQL GIN indexes",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "database:1"


def test_bm25_returns_empty_when_index_is_empty():
    retriever = BM25Retriever()

    results = retriever.search(
        "machine learning",
        top_k=5,
    )

    assert results == []


def test_bm25_preserves_chunk_metadata():
    chunk = DocumentChunk(
        text="Python programming language",
        source="notes.txt",
        page=4,
        chunk_id="notes:4",
    )

    retriever = BM25Retriever()
    retriever.add([chunk])

    results = retriever.search(
        "Python programming",
        top_k=1,
    )

    assert results[0].chunk.source == "notes.txt"
    assert results[0].chunk.page == 4
    assert results[0].chunk.chunk_id == "notes:4"