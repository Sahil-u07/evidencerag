from app.ingestion.models import DocumentChunk
from app.retrieval.dense import DenseRetriever


def test_dense_retriever_returns_highest_similarity_first():
    chunks = [
        DocumentChunk(
            text="machine learning models",
            source="ml.txt",
            chunk_id="ml:1",
        ),
        DocumentChunk(
            text="cooking recipes and ingredients",
            source="food.txt",
            chunk_id="food:1",
        ),
        DocumentChunk(
            text="neural networks and deep learning",
            source="dl.txt",
            chunk_id="dl:1",
        ),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]

    retriever = DenseRetriever()
    retriever.add(chunks, embeddings)

    results = retriever.search(
        query_embedding=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "ml:1"
    assert results[0].score > results[1].score


def test_empty_retriever_returns_no_results():
    retriever = DenseRetriever()

    results = retriever.search(
        query_embedding=[1.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_mismatched_chunks_and_embeddings_raise_error():
    retriever = DenseRetriever()

    chunks = [
        DocumentChunk(
            text="hello",
            source="test.txt",
        )
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    try:
        retriever.add(chunks, embeddings)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "match" in str(exc)