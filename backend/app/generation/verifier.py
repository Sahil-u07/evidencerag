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
    for value, word in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())
    if word != "one"
} | {word: word for word in ("hundred", "thousand", "million")}
NEGATION = re.compile(r"\b(?:not|no|never|none|cannot|without|neither|nor)\b|n't\b", re.IGNORECASE)


@dataclass
class VerificationResult:
    supported: bool
    cited_evidence: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


class EvidenceVerifier:
    """
    Local evidence-grounding verifier using a Natural Language
    Inference (NLI) cross-encoder.

    Verification uses NLI as the primary signal with a strict
    lexical-support fallback for near-verbatim claims. The fallback
    handles cases where the NLI model incorrectly rejects text that
    is directly present in the retrieved evidence.

    Verification flow:

        Generated claim
              ↓
        Cited evidence
              ↓
        NLI classifier
              ↓
        Entailment / Neutral / Contradiction
              ↓
        Strict lexical fallback when necessary

    This is a model-based verification signal, not an absolute
    guarantee of factual correctness.
    """

    CITATION_PATTERN = re.compile(
        r"\[Evidence\s+(\d+)\]",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        entailment_threshold: float = 0.70,
    ) -> None:
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self._nli_model: CrossEncoder | None = None

    @property
    def nli_model(self) -> CrossEncoder:
        """
        Lazily load the local NLI model.

        This avoids loading model weights when only citation
        validation is being used.
        """

        if self._nli_model is None:
            self._nli_model = CrossEncoder(
                self.model_name
            )

        return self._nli_model

    def verify_citations(
        self,
        answer: str,
        evidence_count: int,
    ) -> VerificationResult:
        """
        Validate citation syntax and evidence references.
        """

        if not answer.strip():
            return VerificationResult(
                supported=False,
                reason="Generated answer is empty.",
            )

        if evidence_count <= 0:
            return VerificationResult(
                supported=False,
                reason="No evidence was retrieved.",
            )

        citations = self.CITATION_PATTERN.findall(
            answer
        )

        if not citations:
            return VerificationResult(
                supported=False,
                reason="Answer does not contain an evidence citation.",
            )

        cited_evidence = sorted(
            {
                int(citation)
                for citation in citations
            }
        )

        invalid = [
            evidence_id
            for evidence_id in cited_evidence
            if evidence_id < 1
            or evidence_id > evidence_count
        ]

        if invalid:
            return VerificationResult(
                supported=False,
                cited_evidence=cited_evidence,
                reason=(
                    "Answer contains invalid evidence references: "
                    + ", ".join(
                        map(str, invalid)
                    )
                ),
            )

        return VerificationResult(
            supported=True,
            cited_evidence=cited_evidence,
            reason="Answer contains valid evidence references.",
        )

    def verify_grounding(
        self,
        answer: str,
        evidence: list[Any],
    ) -> VerificationResult:
        """
        Verify every cited claim using NLI.

        NLI is the primary grounding signal. If NLI does not
        recognize a claim that is very strongly supported by the
        cited evidence lexically, a strict lexical fallback is used.
        """

        citation_result = self.verify_citations(
            answer=answer,
            evidence_count=len(evidence),
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
        used_lexical_fallback = False

        for claim, citation_ids in claims:
            claim_supported = False

            for evidence_id in citation_ids:
                chunk = evidence[
                    evidence_id - 1
                ].result.chunk

                evidence_text = chunk.text

                if not self._numbers(claim) <= self._numbers(evidence_text):
                    continue

                if self._is_verbatim_quote(evidence_text, claim):
                    claim_supported = True
                    used_lexical_fallback = True
                    break

                label, confidence = self._predict_nli(
                    premise=evidence_text,
                    hypothesis=claim,
                )

                if label == "entailment" and confidence >= self.entailment_threshold:
                    claim_supported = True
                    break

                # The fallback covers NLI misses, so it must not overrule a contradiction.
                if label != "contradiction" and self._has_strong_lexical_support(evidence_text, claim):
                    claim_supported = True
                    used_lexical_fallback = True
                    break

            if not claim_supported:
                unsupported_claims.append(
                    claim
                )

        if unsupported_claims:
            return VerificationResult(
                supported=False,
                cited_evidence=citation_result.cited_evidence,
                unsupported_claims=unsupported_claims,
                reason=(
                    "One or more cited claims were not "
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
            reason=(
                "All cited claims were supported by "
                "NLI entailment."
            ),
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
        def normalize(text: str) -> str:
            return " ".join(text.split()).casefold()

        return normalize(claim) in normalize(evidence_text) and self._agrees_on_negation(evidence_text, claim)

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
        words = {NUMBER_WORDS[word] for word in re.findall(r"[a-z]+", lowered) if word in NUMBER_WORDS}
        return words | set(re.findall(r"\d+(?:\.\d+)?", lowered))

    @staticmethod
    def _meaningful_tokens(
        text: str,
    ) -> set[str]:
        """
        Normalize text into meaningful lowercase tokens.

        Very common function words are removed so that lexical
        support is based on substantive content rather than words
        such as 'the', 'is', and 'of'.
        """

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

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 2
            and token not in stopwords
        }

    def _predict_nli(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[str, float]:
        """
        Run NLI classification and return:

            (label, confidence)

        Label is normalized to one of:

            contradiction
            entailment
            neutral
        """

        scores = self.nli_model.predict(
            [
                (
                    premise,
                    hypothesis,
                )
            ],
            apply_softmax=True,
        )

        probabilities = np.asarray(
            scores[0],
            dtype=float,
        )

        labels = self._get_model_labels(
            len(probabilities)
        )

        best_index = int(
            np.argmax(probabilities)
        )

        label = labels[best_index]

        confidence = float(
            probabilities[best_index]
        )

        return label, confidence

    def _get_model_labels(
        self,
        number_of_labels: int,
    ) -> list[str]:
        """
        Resolve the model's label ordering from its configuration.

        This avoids hard-coding the numeric order of the NLI classes.
        """

        try:
            config = (
                self.nli_model.model.config
            )

            id2label = getattr(
                config,
                "id2label",
                None,
            )

            if id2label:
                labels = []

                for index in range(
                    number_of_labels
                ):
                    raw_label = id2label.get(
                        index,
                        id2label.get(
                            str(index),
                            "",
                        ),
                    )

                    labels.append(
                        self._normalize_label(
                            str(raw_label)
                        )
                    )

                if all(labels):
                    return labels

        except Exception:
            pass

        # Fallback for the expected 3-class NLI model.
        if number_of_labels == 3:
            return [
                "contradiction",
                "entailment",
                "neutral",
            ]

        return [
            f"label_{index}"
            for index in range(
                number_of_labels
            )
        ]

    def _normalize_label(
        self,
        label: str,
    ) -> str:
        """
        Normalize common NLI label formats.
        """

        normalized = (
            label
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if (
            "entail" in normalized
            or normalized in {
                "label 1",
                "label_1",
            }
        ):
            return "entailment"

        if "contrad" in normalized:
            return "contradiction"

        if "neutral" in normalized:
            return "neutral"

        return normalized.replace(
            " ",
            "_",
        )

    def _extract_claims(self, answer: str) -> list[tuple[str, list[int]]]:
        """Return each sentence with its citations. Uncited sentences get an empty list."""
        claims = []

        for sentence in split_sentences(answer):
            citation_ids = sorted({int(item) for item in self.CITATION_PATTERN.findall(sentence)})
            claim = self.CITATION_PATTERN.sub("", sentence).strip(" .,:;!?\n\t")

            # Replies like "Yes." state no fact of their own.
            if len(self._meaningful_tokens(claim)) < 2 and not citation_ids:
                continue

            if claim:
                claims.append((claim, citation_ids))

        return claims
