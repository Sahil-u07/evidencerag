import pytest

from app.evaluation.evidence import (
    answer_coverage,
    bootstrap_ci,
    relevant_chunk_count,
    relevant_facts,
    score_question,
    validate_cases,
    validate_questions,
)


FACT_A = "the server enters the listening state first"
FACT_B = "the client retries after a short random delay"


def question(**overrides):
    item = {
        "id": "v2-001",
        "split": "dev",
        "type": "paraphrase",
        "query": "What happens first?",
        "evidence": [FACT_A],
    }
    item.update(overrides)
    return item


def test_matching_ignores_case_and_line_wrapping():
    text = "Then The Server\n   enters the  listening\nstate first."
    assert relevant_facts(text, [FACT_A]) == {0}


def test_relevant_chunk_count():
    assert relevant_chunk_count([FACT_A, "other text", FACT_B], [FACT_A, FACT_B]) == 2


def test_score_perfect_single_fact():
    scores = score_question([FACT_A, "x", "y"], [FACT_A], relevant_chunks=1)
    assert scores["hit_at_1"] == 1.0
    assert scores["recall_at_1"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg_at_5"] == pytest.approx(1.0)


def test_capped_recall_reaches_one_when_recall_cannot():
    scores = score_question([FACT_A, FACT_B], [FACT_A, FACT_B], relevant_chunks=2)
    assert scores["recall_at_1"] == 0.5
    assert scores["capped_recall_at_1"] == 1.0
    assert scores["recall_at_3"] == 1.0


def test_score_miss_is_zero_everywhere():
    scores = score_question(["x", "y"], [FACT_A], relevant_chunks=1)
    assert scores["hit_at_5"] == 0.0
    assert scores["mrr"] == 0.0
    assert scores["ndcg_at_5"] == 0.0


def test_score_rank_two():
    scores = score_question(["x", FACT_A], [FACT_A], relevant_chunks=1)
    assert scores["hit_at_1"] == 0.0
    assert scores["hit_at_3"] == 1.0
    assert scores["mrr"] == 0.5


def test_validate_passes_good_question():
    assert validate_questions([question()], [FACT_A]) == []


def test_validate_names_question_and_phrase_when_text_is_missing():
    errors = validate_questions([question()], ["unrelated text"])
    assert len(errors) == 1
    assert "v2-001" in errors[0]
    assert FACT_A in errors[0]
    assert "no chunk" in errors[0]


def test_validate_rejects_phrase_found_in_too_many_chunks():
    errors = validate_questions([question()], [FACT_A] * 4)
    assert "4 chunks" in errors[0]


def test_validate_max_matches_can_be_disabled():
    assert validate_questions([question()], [FACT_A] * 4, max_matches=None) == []


@pytest.mark.parametrize("phrase", ["too short here", " ".join(["word"] * 16)])
def test_validate_rejects_phrase_length(phrase):
    errors = validate_questions([question(evidence=[phrase])], [phrase])
    assert "5 to 15 words" in errors[0]


def test_validate_rejects_unknown_split_and_type():
    errors = validate_questions([question(split="train", type="other")], [FACT_A])
    assert len(errors) == 2


def test_validate_rejects_duplicate_ids_and_phrases():
    errors = validate_questions([question(), question()], [FACT_A])
    assert any("duplicate id" in error for error in errors)
    assert any("duplicate evidence" in error for error in errors)


def test_validate_cases_requires_quoted_passages():
    case = {
        "id": "vc-001",
        "split": "dev",
        "category": "negation",
        "supported": False,
        "answer": "Claim. [Evidence 1]",
        "evidence": ["not in the corpus"],
    }
    errors = validate_cases([case], "some other corpus text")
    assert "vc-001" in errors[0]


def test_bootstrap_ci_is_seeded_and_contains_mean():
    values = [0.0, 1.0] * 20
    assert bootstrap_ci(values) == bootstrap_ci(values)
    low, high = bootstrap_ci(values)
    assert low < 0.5 < high


def test_bootstrap_ci_of_constant_values():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_coverage_averages_the_share_of_words_per_phrase():
    evidence = ["alpha bravo charlie delta", "echo foxtrot golf hotel"]
    answer = "alpha bravo echo foxtrot golf hotel"
    assert answer_coverage(answer, evidence, abstained=False) == pytest.approx(0.75)


def test_coverage_is_zero_for_an_abstained_answer():
    assert answer_coverage("alpha bravo", ["alpha bravo"], abstained=True) == 0.0
