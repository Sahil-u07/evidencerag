import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

RESULT_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "reranker_benchmark_results.json"
)


def first_relevant_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return rank

    return None


def main():
    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    hybrid_queries = {
        item["id"]: item
        for item in data["systems"]["hybrid"]["per_query"]
    }

    reranked_queries = {
        item["id"]: item
        for item in data["systems"]["reranked_hybrid"]["per_query"]
    }

    improved = []
    degraded = []
    unchanged = []

    for query_id, hybrid in hybrid_queries.items():
        reranked = reranked_queries[query_id]

        relevant_ids = hybrid["relevant_ids"]

        hybrid_rank = first_relevant_rank(
            hybrid["retrieved_ids"],
            relevant_ids,
        )

        reranked_rank = first_relevant_rank(
            reranked["retrieved_ids"],
            relevant_ids,
        )

        if hybrid_rank is None and reranked_rank is not None:
            improved.append(
                (query_id, hybrid, reranked)
            )
        elif reranked_rank is None and hybrid_rank is not None:
            degraded.append(
                (query_id, hybrid, reranked)
            )
        elif (
            reranked_rank is not None
            and hybrid_rank is not None
            and reranked_rank < hybrid_rank
        ):
            improved.append(
                (query_id, hybrid, reranked)
            )
        elif (
            hybrid_rank is not None
            and reranked_rank is not None
            and reranked_rank > hybrid_rank
        ):
            degraded.append(
                (query_id, hybrid, reranked)
            )
        else:
            unchanged.append(
                (query_id, hybrid, reranked)
            )

    print("\n" + "=" * 90)
    print("RERANKER DIAGNOSTICS")
    print("=" * 90)

    print(
        f"Improved:  {len(improved)}"
        f"\nDegraded:  {len(degraded)}"
        f"\nUnchanged: {len(unchanged)}"
    )

    print("\n" + "-" * 90)
    print("IMPROVED QUERIES")
    print("-" * 90)

    for query_id, hybrid, reranked in improved:
        print(f"\n[{query_id}]")
        print(f"Query: {hybrid['query']}")
        print(
            f"Hybrid:   {hybrid['retrieved_ids']}"
        )
        print(
            f"Reranked: {reranked['retrieved_ids']}"
        )

    print("\n" + "-" * 90)
    print("DEGRADED QUERIES")
    print("-" * 90)

    for query_id, hybrid, reranked in degraded:
        print(f"\n[{query_id}]")
        print(f"Query: {hybrid['query']}")
        print(
            f"Hybrid:   {hybrid['retrieved_ids']}"
        )
        print(
            f"Reranked: {reranked['retrieved_ids']}"
        )

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()