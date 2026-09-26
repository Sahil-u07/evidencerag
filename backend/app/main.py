from dataclasses import dataclass, field
import re

import numpy as np
from sentence_transformers import CrossEncoder


NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


@dataclass
class VerificationResult:
    supported: bool
    cited_evidence: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


class EvidenceVerifier:
    """
    Local evidence-grounding verifier.

    Verification pipeline:

        Generated claim
              ↓
        Cited evidence
              ↓
        Natural Language Inference
              ↓
        Entailment / Neutral / Contradiction

    The verifier uses a local NLI cross-encoder to determine
    whether the cited evidence entails the generated claim.

    This is still a model-based verification signal and should
    not be treated as an absolute guarantee of factual truth.
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
        Lazy-load the NLI model.

        The model is loaded only when grounding verification
        is actually requested.
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

        citations = self.CITATION_PATTERN.findall(answer)

        if not citations:
            return VerificationResult(
                supported=False,
                reason="Answer does not contain an evidence citation.",
            )

        cited_evidence = sorted(
            {int(citation) for citation in citations}
        )

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

    def verify_grounding(
        self,
        answer: str,
        evidence: list,
    ) -> VerificationResult:
        """
        Verify cited claims using local NLI inference.

        Evidence is treated as the premise and the generated
        claim is treated as the hypothesis.

        A claim is considered grounded only when at least one
        cited evidence block has an entailment score above the
        configured threshold and entailment is the strongest
        NLI class.
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

        for claim, citation_ids in claims:
            claim_supported = False

            for evidence_id in citation_ids:
                chunk = evidence[evidence_id - 1].result.chunk

                evidence_text = chunk.text

                scores = self.nli_model.predict(
                    [
                        (
                            evidence_text,
                            claim,
                        )
                    ],
                    apply_softmax=True,
                )

                probabilities = np.asarray(
                    scores[0],
                    dtype=float,
                )

                # Model label ordering:
                #
                # 0 = contradiction
                # 1 = entailment
                # 2 = neutral
                contradiction_score = float(
                    probabilities[0]
                )

                entailment_score = float(
                    probabilities[1]
                )

                neutral_score = float(
                    probabilities[2]
                )

                strongest_label = int(
                    np.argmax(probabilities)
                )

                if (
                    strongest_label == 1
                    and entailment_score
                    >= self.entailment_threshold
                ):
                    claim_supported = True
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
                    "One or more cited claims were not entailed "
                    "by their cited evidence."
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

    def _extract_claims(
        self,
        answer: str,
    ) -> list[tuple[str, list[int]]]:
        """
        Extract claims and associate citations with them.

        Handles:

            Claim [Evidence 1]

            Claim. [Evidence 1]

            Claim [Evidence 1] [Evidence 2]

            First claim. [Evidence 1]
            Second claim. [Evidence 2]
        """

        normalized = re.sub(
            r"\s+(\[Evidence\s+\d+\])",
            r"\1",
            answer.strip(),
            flags=re.IGNORECASE,
        )

        fragments = re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )

        claims = []

        for fragment in fragments:
            citations = self.CITATION_PATTERN.findall(
                fragment
            )

            if not citations:
                continue

            claim = self.CITATION_PATTERN.sub(
                "",
                fragment,
            ).strip()

            claim = claim.strip(
                " .,:;"
            )

            if not claim:
                continue

            citation_ids = sorted(
                {
                    int(citation)
                    for citation in citations
                }
            )

            claims.append(
                (
                    claim,
                    citation_ids,
                )
            )

        return claims