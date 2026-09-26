from app.ingestion.models import DocumentChunk
from app.retrieval.dense import SearchResult
from app.reranking.cross_encoder import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for query, text in pairs:
            if "database" in text.lower():
                scores.append(0.95)
            elif "python" in text.lower():
                scores.append(0.60)
            else:
                scores.append(0.10)

        return scores


def make_result(
    chunk_id: str,
    text: str,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            text=text,
            source="test.txt",
            chunk_id=chunk_id,
        ),
        score=0.5,
    )


def test_reranker_orders_candidates_by_cross_encoder_score():
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = FakeCrossEncoder()

    results = [
        make_result(
            "A",
            "Python programming language",
        ),
        make_result(
            "B",
            "Database indexing and queries",
        ),
        make_result(
            "C",
            "Cooking recipes",
        ),
    ]

    ranked = reranker.rerank(
        "database search",
        results,
        top_k=3,
    )

    ids = [
        item.result.chunk.chunk_id
        for item in ranked
    ]

    assert ids == ["B", "A", "C"]


def test_reranker_respects_top_k():
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = FakeCrossEncoder()

    results = [
        make_result("A", "Python"),
        make_result("B", "Database"),
        make_result("C", "Cooking"),
    ]

    ranked = reranker.rerank(
        "database",
        results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].result.chunk.chunk_id == "B"


def test_reranker_rejects_empty_query():
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = FakeCrossEncoder()

    try:
        reranker.rerank("", [], top_k=5)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc)