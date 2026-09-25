from dataclasses import dataclass

import numpy as np

from app.ingestion.models import DocumentChunk


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float


class DenseRetriever:
    """In-memory dense vector retrieval using cosine similarity."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        if not chunks:
            return

        matrix = np.asarray(embeddings, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = matrix
        else:
            self.embeddings = np.vstack(
                [self.embeddings, matrix]
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if self.embeddings is None or not self.chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "Query embedding must be a 1D vector"
            )

        scores = self.embeddings @ query

        top_k = min(top_k, len(self.chunks))

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in top_indices
        ]