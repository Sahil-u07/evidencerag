from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.retrieval.dense import SearchResult


@lru_cache(maxsize=4)
def _get_model(
    model_name: str,
) -> CrossEncoder:
    return CrossEncoder(model_name)


@dataclass
class RerankedResult:
    result: SearchResult
    score: float


class CrossEncoderReranker:
    """Rerank retrieved candidates using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = _get_model(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not results:
            return []

        pairs = [
            (query, result.chunk.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            RerankedResult(
                result=result,
                score=float(score),
            )
            for result, score in ranked[:top_k]
        ]