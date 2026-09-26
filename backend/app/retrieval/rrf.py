from collections import defaultdict

from app.retrieval.dense import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 5,
    k: int = 60,
) -> list[SearchResult]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document:
        sum(1 / (k + rank))

    Rank is one-based.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    scores: dict[str, float] = defaultdict(float)
    chunks = {}

    for results in result_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id

            if chunk_id is None:
                raise ValueError(
                    "RRF requires every result to have a chunk_id"
                )

            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            scores[chunk_id] += 1 / (k + rank)
            chunks[chunk_id] = result.chunk

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:top_k]

    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=scores[chunk_id],
        )
        for chunk_id in ranked_ids
    ]