from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.dense import DenseRetriever, SearchResult
from app.retrieval.embedder import TextEmbedder


class IndexService:
    """Coordinates document ingestion, embedding, indexing, and search."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        retriever: DenseRetriever | None = None,
    ) -> None:
        self.embedder = embedder or TextEmbedder()
        self.retriever = retriever or DenseRetriever()

    def index_document(
        self,
        path: str | Path,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> int:
        documents = load_document(path)

        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            return 0

        embeddings = self.embedder.encode(
            [chunk.text for chunk in chunks]
        )

        self.retriever.add(
            chunks,
            embeddings,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self.embedder.encode_query(query)

        return self.retriever.search(
            query_embedding,
            top_k=top_k,
        )