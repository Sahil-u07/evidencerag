from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    supported: bool
    cited_evidence: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


class EvidenceVerifier:
    """
    Verify whether a generated answer has sufficient support
    in the retrieved evidence.

    This class initially provides deterministic validation of
    evidence references. LLM-based claim verification will be
    added separately.
    """

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

        cited_evidence = []

        for evidence_id in range(1, evidence_count + 1):
            marker = f"[Evidence {evidence_id}]"

            if marker in answer:
                cited_evidence.append(evidence_id)

        if not cited_evidence:
            return VerificationResult(
                supported=False,
                reason="Answer does not contain an evidence citation.",
            )

        return VerificationResult(
            supported=True,
            cited_evidence=cited_evidence,
            reason="Answer contains references to retrieved evidence.",
        )