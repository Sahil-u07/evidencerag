import json
from pathlib import Path
from statistics import mean

from app.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever


ROOT_DIR = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT_DIR / "data" / "evaluation" / "corpus"
GOLDEN_SET_PATH = ROOT_DIR / "data" / "evaluation" / "golden_set.json"

RESULT_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "reranker_benchmark_results.json"
)

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

EVALUATION_KS = [1, 3, 5]
NDCG_K = 5

RERANK_CANDIDATE_K = 20


def load_corpus():
    """Load and chunk every document in the evaluation corpus."""
    all_chunks = []

    for file_path in sorted(CORPUS_DIR.glob("*.md")):
        documents = load_document(file_path)

        chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        all_chunks.extend(chunks)

    return all_chunks


def load_golden_set():
    """Load the manually curated evaluation questions."""
    with GOLDEN_SET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_ground_truth(
    golden_set,
    available_ids,
):
    """Ensure every ground-truth chunk ID exists in the corpus."""
    missing = []

    for item in golden_set:
        for relevant_id in item["relevant_ids"]:
            if relevant_id not in available_ids:
                missing.append(
                    f'{item["id"]}: {relevant_id}'
                )

    if missing:
        raise ValueError(
            "Golden set contains unknown chunk IDs:\n"
            + "\n".join(missing)
        )


def average(values):
    """Calculate an arithmetic mean."""
    values = list(values)

    if not values:
        return 0.0

    return mean(values)


def calculate_query_metrics(
    retrieved_ids,
    relevant_ids,
):
    """Calculate retrieval metrics for one query."""
    metrics = {
        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            NDCG_K,
        ),
    }

    for k in EVALUATION_KS:
        metrics[f"recall_at_{k}"] = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        )

    return metrics


def extract_retrieved_ids(
    system_name,
    retrieved,
):
    """
    Normalize the different result object types into chunk IDs.

    Standard retrievers return SearchResult objects:
        result.chunk.chunk_id

    RerankedHybridRetriever returns RerankedResult objects:
        result.result.chunk.chunk_id
    """
    if system_name == "reranked_hybrid":
        return [
            item.result.chunk.chunk_id
            for item in retrieved
        ]

    return [
        item.chunk.chunk_id
        for item in retrieved
    ]


def evaluate_retriever(
    system_name,
    retriever,
    golden_set,
    embedder=None,
):
    """Evaluate one retrieval system over every question."""
    per_query = []

    for item in golden_set:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        if system_name == "dense":
            if embedder is None:
                raise ValueError(
                    "Dense evaluation requires an embedder."
                )

            query_embedding = embedder.encode_query(
                query
            )

            retrieved = retriever.search(
                query_embedding,
                top_k=max(EVALUATION_KS),
            )

        elif system_name == "reranked_hybrid":
            retrieved = retriever.search(
                query,
                top_k=max(EVALUATION_KS),
                candidate_k=RERANK_CANDIDATE_K,
            )

        else:
            retrieved = retriever.search(
                query,
                top_k=max(EVALUATION_KS),
            )

        retrieved_ids = extract_retrieved_ids(
            system_name,
            retrieved,
        )

        metrics = calculate_query_metrics(
            retrieved_ids,
            relevant_ids,
        )

        per_query.append(
            {
                "id": item["id"],
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                **metrics,
            }
        )

    return per_query


def summarize(per_query):
    """Average retrieval metrics across all questions."""
    summary = {}

    for k in EVALUATION_KS:
        summary[f"recall_at_{k}"] = average(
            item[f"recall_at_{k}"]
            for item in per_query
        )

    summary["mrr"] = average(
        item["mrr"]
        for item in per_query
    )

    summary["ndcg_at_5"] = average(
        item["ndcg_at_5"]
        for item in per_query
    )

    return summary


def build_retrievers(
    chunks,
    embeddings,
    embedder,
):
    """Construct all retrieval systems used by the benchmark."""

    # Dense retrieval
    dense = DenseRetriever()
    dense.add(
        chunks,
        embeddings,
    )

    # BM25 retrieval
    bm25 = BM25Retriever()
    bm25.add(chunks)

    # Hybrid retrieval
    hybrid = HybridRetriever(
        embedder=embedder,
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    # Cross-encoder reranking
    print("\nLoading cross-encoder reranker...")

    reranker = CrossEncoderReranker()

    reranked_hybrid = RerankedHybridRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
        candidate_multiplier=4,
        minimum_candidates=20,
    )

    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
        "reranked_hybrid": reranked_hybrid,
    }


def save_results(
    chunks,
    golden_set,
    all_results,
):
    """Persist the full benchmark configuration and results."""
    output = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "evaluation_ks": EVALUATION_KS,
        "ndcg_k": NDCG_K,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "question_count": len(golden_set),
        "chunk_count": len(chunks),
        "systems": all_results,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )


def print_summary(all_results):
    """Print a compact comparison table."""
    print("\n" + "=" * 90)
    print("RETRIEVAL BENCHMARK — RERANKER")
    print("=" * 90)

    print(
        f"{'System':<20}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
    )

    for system_name, result in all_results.items():
        metrics = result["average"]

        print(
            f"{system_name:<20}"
            f"{metrics['recall_at_1']:>10.4f}"
            f"{metrics['recall_at_3']:>10.4f}"
            f"{metrics['recall_at_5']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['ndcg_at_5']:>12.4f}"
        )

    print("=" * 90)


def main():
    print("Loading evaluation corpus...")

    chunks = load_corpus()
    golden_set = load_golden_set()

    available_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    validate_ground_truth(
        golden_set,
        available_ids,
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Questions: {len(golden_set)}")
    print(
        f"Chunk size: {CHUNK_SIZE} words | "
        f"overlap: {CHUNK_OVERLAP} words"
    )

    # ---------------------------------------------------------
    # Embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedder = TextEmbedder()

    print("Generating document embeddings...")

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

    # ---------------------------------------------------------
    # Build retrieval systems
    # ---------------------------------------------------------

    systems = build_retrievers(
        chunks,
        embeddings,
        embedder,
    )

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------

    all_results = {}

    for system_name, retriever in systems.items():
        print(
            f"\nEvaluating {system_name}..."
        )

        per_query = evaluate_retriever(
            system_name=system_name,
            retriever=retriever,
            golden_set=golden_set,
            embedder=embedder,
        )

        all_results[system_name] = {
            "per_query": per_query,
            "average": summarize(per_query),
        }

    # ---------------------------------------------------------
    # Save and display
    # ---------------------------------------------------------

    save_results(
        chunks,
        golden_set,
        all_results,
    )

    print_summary(all_results)

    print(
        f"Results saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()