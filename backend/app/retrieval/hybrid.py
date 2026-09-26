from app.ingestion.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever, SearchResult
from app.retrieval.embedder import TextEmbedder
from app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Combine semantic and lexical retrieval using RRF."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        fusion_k: int = 60,
    ) -> None:
        if fusion_k <= 0:
            raise ValueError("fusion_k must be greater than 0")

        self.embedder = embedder or TextEmbedder()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.fusion_k = fusion_k

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add the same chunks to both retrieval systems."""
        self.dense_retriever.add(chunks, embeddings)
        self.bm25_retriever.add(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve and fuse semantic and lexical results."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_k = candidate_k or max(top_k * 5, 10)

        query_embedding = self.embedder.encode_query(query)

        dense_results = self.dense_retriever.search(
            query_embedding,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        return reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=top_k,
            k=self.fusion_k,
        )