import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.dense import SearchResult


def tokenize(text: str) -> list[str]:
    """Convert text into normalized tokens for lexical retrieval."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Lexical retrieval using the BM25 ranking algorithm."""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.index: BM25Okapi | None = None

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add document chunks and rebuild the BM25 index."""
        if not chunks:
            return

        self.chunks.extend(chunks)

        tokenized_documents = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.index = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the highest-scoring lexical matches."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.index is None or not self.chunks:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        top_k = min(top_k, len(self.chunks))
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in top_indices
        ]