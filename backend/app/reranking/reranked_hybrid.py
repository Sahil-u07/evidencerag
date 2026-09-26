from app.reranking.cross_encoder import (
    CrossEncoderReranker,
    RerankedResult,
)
from app.retrieval.hybrid import HybridRetriever


class RerankedHybridRetriever:
    """
    Hybrid retrieval followed by cross-encoder reranking.

    Stage 1:
        Dense + BM25 + RRF

    Stage 2:
        Cross-encoder reranking
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        candidate_multiplier: int = 4,
        minimum_candidates: int = 20,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than 0"
            )

        if minimum_candidates <= 0:
            raise ValueError(
                "minimum_candidates must be greater than 0"
            )

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.minimum_candidates = minimum_candidates

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if candidate_k is None:
            candidate_k = max(
                top_k * self.candidate_multiplier,
                self.minimum_candidates,
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        candidates = self.hybrid_retriever.search(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )