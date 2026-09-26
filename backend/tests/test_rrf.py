from app.ingestion.models import DocumentChunk
from app.retrieval.dense import SearchResult
from app.retrieval.rrf import reciprocal_rank_fusion


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            text=f"content for {chunk_id}",
            source="test.txt",
            chunk_id=chunk_id,
        ),
        score=score,
    )


def test_rrf_combines_rankings():
    dense_results = [
        make_result("A", 0.99),
        make_result("B", 0.90),
        make_result("C", 0.80),
    ]

    bm25_results = [
        make_result("B", 12.0),
        make_result("C", 8.0),
        make_result("A", 6.0),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        top_k=3,
        k=1,
    )

    ids = [result.chunk.chunk_id for result in results]

    assert ids == ["B", "A", "C"]


def test_rrf_includes_results_found_by_only_one_retriever():
    dense_results = [
        make_result("A", 0.99),
    ]

    bm25_results = [
        make_result("B", 10.0),
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        top_k=2,
        k=1,
    )

    ids = {result.chunk.chunk_id for result in results}

    assert ids == {"A", "B"}


def test_rrf_handles_duplicate_result_in_same_list():
    results = reciprocal_rank_fusion(
        [
            [
                make_result("A", 1.0),
                make_result("A", 0.9),
                make_result("B", 0.8),
            ]
        ],
        top_k=2,
        k=1,
    )

    ids = [result.chunk.chunk_id for result in results]

    assert ids == ["A", "B"]


def test_rrf_rejects_results_without_chunk_id():
    result = SearchResult(
        chunk=DocumentChunk(
            text="missing id",
            source="test.txt",
        ),
        score=1.0,
    )

    try:
        reciprocal_rank_fusion([[result]])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "chunk_id" in str(exc)