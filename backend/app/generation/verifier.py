from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.text import split_sentences


NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# "one" is left out because it is mostly a pronoun.
NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
    if word != "one"
} | {word: word for word in ("hundred", "thousand", "million")}
NEGATION = re.compile(
    r"\b(?:not|no|never|none|cannot|without|neither|nor|unless|except|only if|only when)\b|n['’]t\b",
    re.IGNORECASE,
)


@dataclass
class VerificationResult:
    supported: bool
    cited_evidence: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


class EvidenceVerifier:
    """Check cited claims against their evidence with NLI and a strict lexical fallback.

    This is a model-based signal, not a guarantee of factual correctness.
    """

    CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]", re.IGNORECASE)

    def __init__(
        self, model_name: str = NLI_MODEL, entailment_threshold: float = 0.70
    ) -> None:
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self._nli_model: CrossEncoder | None = None

    @property
    def nli_model(self) -> CrossEncoder:
        """Loaded on first use, because citation checks alone do not need it."""

        if self._nli_model is None:
            self._nli_model = CrossEncoder(self.model_name)

        return self._nli_model

    def verify_citations(self, answer: str, evidence_count: int) -> VerificationResult:
        if not answer.strip():
            return VerificationResult(
                supported=False, reason="Generated answer is empty."
            )

        if evidence_count <= 0:
            return VerificationResult(
                supported=False, reason="No evidence was retrieved."
            )

        citations = self.CITATION_PATTERN.findall(answer)

        if not citations:
            return VerificationResult(
                supported=False, reason="Answer does not contain an evidence citation."
            )

        cited_evidence = sorted({int(citation) for citation in citations})

        invalid = [
            evidence_id
            for evidence_id in cited_evidence
            if evidence_id < 1 or evidence_id > evidence_count
        ]

        if invalid:
            return VerificationResult(
                supported=False,
                cited_evidence=cited_evidence,
                reason=(
                    "Answer contains invalid evidence references: "
                    + ", ".join(map(str, invalid))
                ),
            )

        return VerificationResult(
            supported=True,
            cited_evidence=cited_evidence,
            reason="Answer contains valid evidence references.",
        )

    def verify_grounding(self, answer: str, evidence: list[Any]) -> VerificationResult:
        """Verify every cited claim, using NLI first and the lexical fallback second."""

        citation_result = self.verify_citations(
            answer=answer, evidence_count=len(evidence)
        )

        if not citation_result.supported:
            return citation_result

        claims = self._extract_claims(answer)

        if not claims:
            return VerificationResult(
                supported=False,
                cited_evidence=citation_result.cited_evidence,
                reason="No cited claims could be extracted.",
            )

        unsupported_claims = []
        has_uncited = False
        used_lexical_fallback = False

        for claim, citation_ids in claims:
            claim_supported = False

            for evidence_id in citation_ids:
                chunk = evidence[evidence_id - 1].result.chunk

                evidence_text = chunk.text

                if not self._numbers(claim) <= self._numbers(evidence_text):
                    continue

                if self._is_verbatim_quote(evidence_text, claim):
                    claim_supported = True
                    used_lexical_fallback = True
                    break

                label, confidence = self._predict_nli(
                    premise=evidence_text, hypothesis=claim
                )

                if label == "entailment" and confidence >= self.entailment_threshold:
                    claim_supported = True
                    break

                # The fallback covers NLI misses, so it must not overrule a contradiction.
                if label != "contradiction" and self._has_strong_lexical_support(
                    evidence_text, claim
                ):
                    claim_supported = True
                    used_lexical_fallback = True
                    break

            if not claim_supported:
                unsupported_claims.append(claim)
                has_uncited = has_uncited or not citation_ids

        if unsupported_claims:
            return VerificationResult(
                supported=False,
                cited_evidence=citation_result.cited_evidence,
                unsupported_claims=unsupported_claims,
                reason=(
                    "One or more sentences were uncited or "
                    "not supported by their cited evidence."
                    if has_uncited
                    else "One or more cited claims were not "
                    "entailed by their cited evidence."
                ),
            )

        if used_lexical_fallback:
            return VerificationResult(
                supported=True,
                cited_evidence=citation_result.cited_evidence,
                reason=(
                    "All cited claims were supported by NLI "
                    "or strong lexical evidence overlap."
                ),
            )

        return VerificationResult(
            supported=True,
            cited_evidence=citation_result.cited_evidence,
            reason=("All cited claims were supported by NLI entailment."),
        )

    def _has_strong_lexical_support(self, evidence_text: str, claim: str) -> bool:
        """Accept near-verbatim claims that the NLI model failed to recognize."""
        evidence_tokens = self._meaningful_tokens(evidence_text)
        claim_tokens = self._meaningful_tokens(claim)

        if len(claim_tokens) < 5 or not evidence_tokens:
            return False

        overlap = claim_tokens & evidence_tokens

        if len(overlap) < 5 or len(overlap) / len(claim_tokens) < 0.75:
            return False

        return self._agrees_on_negation(evidence_text, claim)

    def _is_verbatim_quote(self, evidence_text: str, claim: str) -> bool:
        def words(text: str) -> str:
            return " " + " ".join(re.findall(r"[a-z0-9]+", text.casefold())) + " "

        # Short quotes are left to NLI: they are often true only in context.
        if len(self._meaningful_tokens(claim)) < 5:
            return False

        return words(claim) in words(evidence_text) and self._agrees_on_negation(
            evidence_text, claim
        )

    def _agrees_on_negation(self, evidence_text: str, claim: str) -> bool:
        claim_tokens = self._meaningful_tokens(claim)
        closest = max(
            split_sentences(evidence_text),
            key=lambda sentence: len(self._meaningful_tokens(sentence) & claim_tokens),
        )

        return bool(NEGATION.search(claim)) == bool(NEGATION.search(closest))

    @staticmethod
    def _numbers(text: str) -> set[str]:
        lowered = text.lower()
        words = {
            NUMBER_WORDS[word]
            for word in re.findall(r"[a-z]+", lowered)
            if word in NUMBER_WORDS
        }
        return words | set(re.findall(r"\d+(?:\.\d+)?", lowered))

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        """Lowercase content words, so lexical support ignores function words."""

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "as",
            "at",
            "by",
            "from",
            "with",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "their",
            "they",
            "them",
            "can",
            "may",
            "will",
            "used",
            "use",
        }

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

        return {token for token in tokens if len(token) > 2 and token not in stopwords}

    def _predict_nli(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Return the model's top label in lowercase, and its confidence."""

        scores = self.nli_model.predict([(premise, hypothesis)], apply_softmax=True)

        probabilities = np.asarray(scores[0], dtype=float)

        best_index = int(np.argmax(probabilities))

        # Class order differs between NLI models, so read it from the config.
        label = self.nli_model.model.config.id2label[best_index].lower()

        confidence = float(probabilities[best_index])

        return label, confidence

    def _extract_claims(self, answer: str) -> list[tuple[str, list[int]]]:
        """Return each sentence with its citations. Uncited sentences get an empty list."""
        claims = []

        for sentence in split_sentences(answer):
            citation_ids = sorted(
                {int(item) for item in self.CITATION_PATTERN.findall(sentence)}
            )
            claim = self.CITATION_PATTERN.sub("", sentence).strip(" .,:;!?\n\t")
            claim = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", claim)

            # Replies like "Yes." state no fact of their own.
            if (
                len(self._meaningful_tokens(claim)) < 2
                and not citation_ids
                and not self._numbers(claim)
            ):
                continue

            if claim:
                claims.append((claim, citation_ids))

        return claims
