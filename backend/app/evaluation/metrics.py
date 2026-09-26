import math
from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than 0")


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Calculate Recall@K for binary relevance."""
    _validate_k(k)

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
) -> float:
    """Calculate Reciprocal Rank."""
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Calculate binary-relevance nDCG@K."""
    _validate_k(k)

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        relevance = 1.0 if chunk_id in relevant else 0.0
        dcg += relevance / math.log2(rank + 1)

    ideal_relevance_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevance_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate the main retrieval metrics for one query."""
    return {
        "recall_at_k": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
        "ndcg_at_k": ndcg_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
    }