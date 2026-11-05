import re

from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.dense import SearchResult


STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in is it its of on or "
    "that the this to was were what when which why with".split()
)


def _stem(token: str) -> str:
    if token.isalpha() and len(token) > 4:
        return re.sub(r"(ing|ed|es|s)$", "", token)
    return token


def tokenize(text: str) -> list[str]:
    """Convert text into normalized tokens for lexical retrieval."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [_stem(token) for token in tokens if token not in STOPWORDS]


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