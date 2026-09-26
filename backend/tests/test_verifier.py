from app.generation.verifier import EvidenceVerifier


def test_supported_answer_with_citation():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists. [Evidence 1]",
        evidence_count=3,
    )

    assert result.supported is True
    assert result.cited_evidence == [1]


def test_supported_answer_with_multiple_citations():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists [Evidence 1] and "
        "reranking improves candidate ordering [Evidence 2].",
        evidence_count=3,
    )

    assert result.supported is True
    assert result.cited_evidence == [1, 2]


def test_answer_without_citation_is_unsupported():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists.",
        evidence_count=3,
    )

    assert result.supported is False
    assert result.cited_evidence == []


def test_no_evidence_is_unsupported():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists. [Evidence 1]",
        evidence_count=0,
    )

    assert result.supported is False


def test_empty_answer_is_unsupported():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "",
        evidence_count=3,
    )

    assert result.supported is False