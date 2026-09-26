from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever


class FakeEmbedder:
    def encode_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text="Python programming language",
            source="python.txt",
            chunk_id="A",
        ),
        DocumentChunk(
            text="Python database indexing",
            source="database.txt",
            chunk_id="B",
        ),
        DocumentChunk(
            text="Cooking recipes and ingredients",
            source="food.txt",
            chunk_id="C",
        ),
    ]


def test_hybrid_combines_dense_and_bm25():
    chunks = make_chunks()

    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
        fusion_k=1,
    )

    retriever.add(chunks, embeddings)

    results = retriever.search(
        "Python database",
        top_k=2,
    )

    ids = [result.chunk.chunk_id for result in results]

    assert len(results) == 2
    assert "A" in ids
    assert "B" in ids
    assert "C" not in ids


def test_hybrid_respects_top_k():
    chunks = make_chunks()

    embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
    )

    retriever.add(chunks, embeddings)

    results = retriever.search(
        "Python",
        top_k=1,
    )

    assert len(results) == 1


def test_hybrid_rejects_empty_query():
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        dense_retriever=DenseRetriever(),
        bm25_retriever=BM25Retriever(),
    )

    try:
        retriever.search("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc)