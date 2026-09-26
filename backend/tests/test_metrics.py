import pytest

from app.evaluation.metrics import (
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"B", "D"}

    result = recall_at_k(
        retrieved,
        relevant,
        k=3,
    )

    assert result == pytest.approx(0.5)


def test_recall_at_k_can_reach_one():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    result = recall_at_k(
        retrieved,
        relevant,
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = {"B"}

    result = reciprocal_rank(
        retrieved,
        relevant,
    )

    assert result == pytest.approx(0.5)


def test_ndcg_perfect_ranking():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "B"}

    result = ndcg_at_k(
        retrieved,
        relevant,
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_evaluate_ranking_returns_all_metrics():
    retrieved = ["X", "A", "B", "Y"]
    relevant = {"A", "B"}

    result = evaluate_ranking(
        retrieved,
        relevant,
        k=4,
    )

    assert set(result.keys()) == {
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
    }

    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(0.5)
    assert 0.0 < result["ndcg_at_k"] < 1.0