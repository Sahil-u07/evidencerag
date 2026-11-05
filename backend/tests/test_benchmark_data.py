import json
from collections import Counter

from app.evaluation.evidence import validate_cases, validate_questions
from scripts import benchmark


QUESTIONS = json.loads(benchmark.QUESTIONS_PATH.read_text(encoding="utf-8"))
CASES = json.loads(benchmark.CASES_PATH.read_text(encoding="utf-8"))

SHIPPED_CHUNKING = benchmark.RetrievalConfig(chunking="words", chunk_size=50, chunk_overlap=10)


def test_question_labels_match_the_corpus():
    texts = [c.text for c in benchmark.chunk_corpus(benchmark.v2_corpus_paths(), SHIPPED_CHUNKING)]
    assert validate_questions(QUESTIONS, texts) == []


def test_question_split_and_type_balance():
    counts = Counter((item["split"], item["type"]) for item in QUESTIONS)
    for split in ("dev", "test"):
        assert counts[(split, "paraphrase")] == 20
        assert counts[(split, "identifier")] == 12
        assert counts[(split, "multi_fact")] == 8


def test_multi_fact_questions_have_two_facts():
    for item in QUESTIONS:
        expected = 2 if item["type"] == "multi_fact" else 1
        assert len(item["evidence"]) == expected, item["id"]


def test_case_passages_are_quoted_from_the_corpus():
    assert validate_cases(CASES, benchmark.read_corpus_text()) == []


def test_case_split_and_category_balance():
    assert Counter(case["split"] for case in CASES) == {"dev": 30, "test": 30}
    categories = {
        True: {"verbatim", "paraphrase", "multi_sentence", "abbreviation_or_decimal"},
        False: {"negation", "wrong_number", "wrong_entity", "uncited", "wrong_citation"},
    }
    for case in CASES:
        assert case["category"] in categories[case["supported"]], case["id"]
    for split in ("dev", "test"):
        seen = {case["category"] for case in CASES if case["split"] == split}
        assert seen == categories[True] | categories[False]
