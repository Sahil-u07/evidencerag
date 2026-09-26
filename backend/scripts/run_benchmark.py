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
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever


ROOT_DIR = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT_DIR / "data" / "evaluation" / "corpus"
GOLDEN_SET_PATH = ROOT_DIR / "data" / "evaluation" / "golden_set.json"
RESULT_PATH = ROOT_DIR / "data" / "evaluation" / "benchmark_results.json"

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

EVALUATION_KS = [1, 3, 5]
NDCG_K = 5


def load_corpus():
    """Load all evaluation documents and convert them into chunks."""
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
    """Load the manually defined evaluation questions."""
    with GOLDEN_SET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_ground_truth(golden_set, available_ids):
    """Ensure every ground-truth chunk actually exists."""
    missing = []

    for item in golden_set:
        for relevant_id in item["relevant_ids"]:
            if relevant_id not in available_ids:
                missing.append(
                    f'{item["id"]}: {relevant_id}'
                )

    if missing:
        details = "\n".join(missing)

        raise ValueError(
            "Golden set contains unknown chunk IDs:\n"
            + details
        )


def average(values):
    """Return the arithmetic mean of a list of values."""
    if not values:
        return 0.0

    return mean(values)


def evaluate_system(
    system_name,
    retriever,
    golden_set,
    embedder=None,
):
    """Evaluate one retrieval system over the complete golden set."""
    per_query = []

    for item in golden_set:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        if system_name == "dense":
            if embedder is None:
                raise ValueError(
                    "Dense evaluation requires an embedder."
                )

            query_embedding = embedder.encode_query(query)

            retrieved = retriever.search(
                query_embedding,
                top_k=max(EVALUATION_KS),
            )

        else:
            retrieved = retriever.search(
                query,
                top_k=max(EVALUATION_KS),
            )

        retrieved_ids = [
            result.chunk.chunk_id
            for result in retrieved
        ]

        query_metrics = {
            "id": item["id"],
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "reciprocal_rank": reciprocal_rank(
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
            query_metrics[f"recall_at_{k}"] = recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )

        per_query.append(query_metrics)

    return per_query


def summarize_results(per_query):
    """Calculate average metrics over all evaluation questions."""
    summary = {}

    for k in EVALUATION_KS:
        summary[f"recall_at_{k}"] = average(
            item[f"recall_at_{k}"]
            for item in per_query
        )

    summary["mrr"] = average(
        item["reciprocal_rank"]
        for item in per_query
    )

    summary["ndcg_at_5"] = average(
        item["ndcg_at_5"]
        for item in per_query
    )

    return summary


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

    print("\nLoading embedding model...")

    embedder = TextEmbedder()

    print("Generating document embeddings...")

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

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

    systems = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    all_results = {}

    for system_name, retriever in systems.items():
        print(f"\nEvaluating {system_name}...")

        per_query = evaluate_system(
            system_name=system_name,
            retriever=retriever,
            golden_set=golden_set,
            embedder=embedder,
        )

        all_results[system_name] = {
            "per_query": per_query,
            "average": summarize_results(per_query),
        }

    output = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "evaluation_ks": EVALUATION_KS,
        "ndcg_k": NDCG_K,
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

    print("\n" + "=" * 78)
    print("RETRIEVAL BENCHMARK V2")
    print("=" * 78)

    print(
        f"{'System':<12}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>10}"
    )

    for system_name, result in all_results.items():
        metrics = result["average"]

        print(
            f"{system_name:<12}"
            f"{metrics['recall_at_1']:>10.4f}"
            f"{metrics['recall_at_3']:>10.4f}"
            f"{metrics['recall_at_5']:>10.4f}"
            f"{metrics['mrr']:>10.4f}"
            f"{metrics['ndcg_at_5']:>10.4f}"
        )

    print("=" * 78)

    print(
        f"Results saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()