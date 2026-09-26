from app.ingestion.models import DocumentChunk
from app.reranking.cross_encoder import RerankedResult
from app.reranking.reranked_hybrid import RerankedHybridRetriever
from app.retrieval.dense import SearchResult


class FakeHybridRetriever:
    def __init__(self):
        self.received_candidate_k = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        self.received_candidate_k = candidate_k

        return [
            SearchResult(
                chunk=DocumentChunk(
                    text="Python language",
                    source="python.txt",
                    chunk_id="A",
                ),
                score=0.90,
            ),
            SearchResult(
                chunk=DocumentChunk(
                    text="Database indexing",
                    source="database.txt",
                    chunk_id="B",
                ),
                score=0.80,
            ),
            SearchResult(
                chunk=DocumentChunk(
                    text="Machine learning",
                    source="ml.txt",
                    chunk_id="C",
                ),
                score=0.70,
            ),
        ][:top_k]


class FakeReranker:
    def __init__(self):
        self.received_candidates = None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        self.received_candidates = results

        ranked = sorted(
            results,
            key=lambda result: result.chunk.text.lower(),
        )

        return [
            RerankedResult(
                result=result,
                score=1.0 - index * 0.1,
            )
            for index, result in enumerate(ranked[:top_k])
        ]


def test_reranked_hybrid_retrieves_candidates_before_reranking():
    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    results = retriever.search(
        "database",
        top_k=5,
    )

    assert hybrid.received_candidate_k == 20
    assert reranker.received_candidates is not None
    assert len(results) == 3


def test_reranked_hybrid_respects_explicit_candidate_k():
    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    retriever.search(
        "database",
        top_k=2,
        candidate_k=10,
    )

    assert hybrid.received_candidate_k == 10


def test_reranked_hybrid_rejects_invalid_candidate_k():
    hybrid = FakeHybridRetriever()
    reranker = FakeReranker()

    retriever = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    try:
        retriever.search(
            "database",
            top_k=5,
            candidate_k=3,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "candidate_k" in str(exc)