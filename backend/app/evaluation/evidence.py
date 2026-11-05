import math
from collections import Counter

import numpy as np


QUESTION_SPLITS = {"dev", "test"}
QUESTION_TYPES = {"paraphrase", "identifier", "multi_fact"}


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def relevant_facts(chunk_text: str, evidence: list[str]) -> set[int]:
    normalized = normalize_text(chunk_text)
    return {
        index
        for index, phrase in enumerate(evidence)
        if normalize_text(phrase) in normalized
    }


def relevant_chunk_count(chunk_texts: list[str], evidence: list[str]) -> int:
    return sum(1 for text in chunk_texts if relevant_facts(text, evidence))


def score_question(
    retrieved_texts: list[str],
    evidence: list[str],
    relevant_chunks: int,
    ks: tuple[int, ...] = (1, 3, 5),
    ndcg_k: int = 5,
) -> dict[str, float]:
    found = [relevant_facts(text, evidence) for text in retrieved_texts]
    scores: dict[str, float] = {}

    for k in ks:
        covered = set().union(*found[:k])
        scores[f"hit_at_{k}"] = 1.0 if covered else 0.0
        scores[f"recall_at_{k}"] = len(covered) / len(evidence)
        scores[f"capped_recall_at_{k}"] = len(covered) / min(k, len(evidence))

    first_rank = next((rank for rank, facts in enumerate(found, 1) if facts), None)
    scores["mrr"] = 1.0 / first_rank if first_rank else 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, facts in enumerate(found[:ndcg_k], 1)
        if facts
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(ndcg_k, relevant_chunks) + 1)
    )
    scores[f"ndcg_at_{ndcg_k}"] = dcg / ideal if ideal else 0.0

    return scores


def validate_questions(
    questions: list[dict],
    chunk_texts: list[str],
    max_matches: int | None = 3,
) -> list[str]:
    errors: list[str] = []
    normalized_chunks = [normalize_text(text) for text in chunk_texts]

    for name, count in Counter(item["id"] for item in questions).items():
        if count > 1:
            errors.append(f"{name}: duplicate id")

    seen_phrases: set[str] = set()

    for item in questions:
        if item["split"] not in QUESTION_SPLITS:
            errors.append(f"{item['id']}: unknown split {item['split']}")
        if item["type"] not in QUESTION_TYPES:
            errors.append(f"{item['id']}: unknown type {item['type']}")

        for phrase in item["evidence"]:
            normalized = normalize_text(phrase)

            if not 5 <= len(normalized.split()) <= 15:
                errors.append(f"{item['id']}: '{phrase}' must be 5 to 15 words")
                continue

            if normalized in seen_phrases:
                errors.append(f"{item['id']}: duplicate evidence '{phrase}'")
            seen_phrases.add(normalized)

            matches = sum(1 for text in normalized_chunks if normalized in text)

            if matches == 0:
                errors.append(f"{item['id']}: '{phrase}' found in no chunk")
            elif max_matches is not None and matches > max_matches:
                errors.append(f"{item['id']}: '{phrase}' found in {matches} chunks")

    return errors


def validate_cases(cases: list[dict], corpus_text: str) -> list[str]:
    errors: list[str] = []
    corpus = normalize_text(corpus_text)

    for case in cases:
        for passage in case["evidence"]:
            if normalize_text(passage) not in corpus:
                errors.append(f"{case['id']}: passage not in corpus: '{passage[:60]}'")

    return errors


def bootstrap_ci(
    values: list[float],
    resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    picks = np.random.default_rng(seed).integers(0, data.size, (resamples, data.size))
    means = data[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
