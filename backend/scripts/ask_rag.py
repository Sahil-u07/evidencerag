from pathlib import Path

from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever


DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "corpus"
)


def build_retriever() -> RerankedHybridRetriever:
    embedder = TextEmbedder()

    documents = []

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue

        documents.extend(load_document(path))

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        overlap=10,
    )

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

    dense = DenseRetriever()
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(
        embedder=embedder,
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    hybrid.add(
        chunks,
        embeddings,
    )

    reranker = CrossEncoderReranker()

    return RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )


def main() -> None:
    print("Building EvidenceRAG retrieval index...")

    retriever = build_retriever()

    generator = OllamaGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    query = "What is reciprocal rank fusion and why is it useful?"

    print("\nRunning local RAG query...")

    result = pipeline.ask(
        query,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(result.answer.answer)

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for index, evidence in enumerate(result.evidence, start=1):
        chunk = evidence.result.chunk

        print(f"\n[Evidence {index}]")
        print(f"Source: {chunk.source}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Reranker score: {evidence.score:.4f}")
        print(f"Text: {chunk.text}")


if __name__ == "__main__":
    main()