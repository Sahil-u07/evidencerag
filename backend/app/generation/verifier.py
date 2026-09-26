from dataclasses import dataclass, field
import re


@dataclass
class VerificationResult:
    supported: bool
    cited_evidence: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


class EvidenceVerifier:
    """
    Lightweight deterministic grounding verifier.

    Responsibilities:
    1. Validate evidence citations.
    2. Associate citations with the claims they support.
    3. Check whether cited claims have meaningful textual
       support in the cited evidence.

    This is a deterministic lexical grounding check.
    It is NOT a semantic entailment model.
    """

    CITATION_PATTERN = re.compile(
        r"\[Evidence\s+(\d+)\]",
        re.IGNORECASE,
    )

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "used",
        "using",
        "was",
        "were",
        "what",
        "which",
        "why",
        "will",
        "with",
        "would",
    }

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
        min_overlap: int = 2,
        min_coverage: float = 0.5,
    ) -> VerificationResult:
        """
        Verify that every cited claim has sufficient textual
        support in its cited evidence.

        A claim must satisfy BOTH:
        - minimum meaningful-token overlap
        - minimum claim-token coverage
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
            claim_tokens = self._keywords(claim)

            if not claim_tokens:
                unsupported_claims.append(claim)
                continue

            claim_supported = False

            for evidence_id in citation_ids:
                chunk = evidence[evidence_id - 1].result.chunk

                evidence_tokens = self._keywords(
                    chunk.text
                )

                overlap = claim_tokens & evidence_tokens

                coverage = (
                    len(overlap) / len(claim_tokens)
                    if claim_tokens
                    else 0.0
                )

                if (
                    len(overlap) >= min_overlap
                    and coverage >= min_coverage
                ):
                    claim_supported = True
                    break

            if not claim_supported:
                unsupported_claims.append(claim)

        if unsupported_claims:
            return VerificationResult(
                supported=False,
                cited_evidence=citation_result.cited_evidence,
                unsupported_claims=unsupported_claims,
                reason=(
                    "One or more cited claims lack sufficient "
                    "textual support in their cited evidence."
                ),
            )

        return VerificationResult(
            supported=True,
            cited_evidence=citation_result.cited_evidence,
            reason=(
                "All cited claims have sufficient textual support "
                "in their cited evidence."
            ),
        )

    def _extract_claims(
        self,
        answer: str,
    ) -> list[tuple[str, list[int]]]:
        """
        Extract claims and attach citations to the sentence
        immediately preceding them.

        Supported formats:

            Claim [Evidence 1]

            Claim. [Evidence 1]

            Claim [Evidence 1] [Evidence 2]

            First claim. [Evidence 1]
            Second claim. [Evidence 2]
        """

        # Critical normalization:
        #
        # "Claim. [Evidence 1]"
        #
        # becomes:
        #
        # "Claim.[Evidence 1]"
        #
        # This keeps the citation attached to the claim when
        # sentence boundaries are detected.
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
                {int(citation) for citation in citations}
            )

            claims.append(
                (
                    claim,
                    citation_ids,
                )
            )

        return claims

    def _keywords(self, text: str) -> set[str]:
        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) >= 3
            and token not in self.STOPWORDS
        }