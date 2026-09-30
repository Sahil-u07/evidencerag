from types import SimpleNamespace

from app.generation.verifier import EvidenceVerifier


def make_evidence(text):
    chunk = SimpleNamespace(
        text=text,
        source="information_retrieval.md",
        page=None,
        chunk_id="ir:1",
    )

    search_result = SimpleNamespace(
        chunk=chunk,
        score=0.95,
    )

    return SimpleNamespace(
        result=search_result,
        score=0.95,
    )


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


def test_invalid_citation_reference_is_unsupported():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists. [Evidence 7]",
        evidence_count=3,
    )

    assert result.supported is False
    assert result.cited_evidence == [7]


def test_citation_normalization_is_case_insensitive():
    verifier = EvidenceVerifier()

    result = verifier.verify_citations(
        "RRF combines ranked lists. [evidence 1]",
        evidence_count=3,
    )

    assert result.supported is True
    assert result.cited_evidence == [1]


def test_grounding_accepts_supported_claim():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "Reciprocal Rank Fusion combines ranked retrieval results "
            "from multiple retrieval systems."
        )
    ]

    result = verifier.verify_grounding(
        "RRF combines ranked retrieval results. [Evidence 1]",
        evidence,
    )

    assert result.supported is True
    assert result.cited_evidence == [1]
    assert result.unsupported_claims == []


def test_grounding_rejects_claim_with_no_supported_terms():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "Reciprocal Rank Fusion combines ranked retrieval results "
            "from multiple retrieval systems."
        )
    ]

    result = verifier.verify_grounding(
        "Quantum computers use superconducting qubits. [Evidence 1]",
        evidence,
    )

    assert result.supported is False
    assert result.cited_evidence == [1]
    assert len(result.unsupported_claims) == 1


def test_grounding_rejects_unrelated_claim():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a lexical retrieval algorithm based on term "
            "frequency and inverse document frequency."
        )
    ]

    result = verifier.verify_grounding(
        "Neural networks require labeled image datasets. [Evidence 1]",
        evidence,
    )

    assert result.supported is False
    assert result.cited_evidence == [1]
    assert len(result.unsupported_claims) == 1


def test_grounding_rejects_mixed_supported_and_unsupported_claims():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a probabilistic lexical ranking function that "
            "considers query-term frequency, document length, and "
            "the rarity of terms across the collection."
        )
    ]

    result = verifier.verify_grounding(
        "BM25 is a probabilistic lexical ranking function. [Evidence 1] "
        "BM25 was invented by Google. [Evidence 1]",
        evidence,
    )

    assert result.supported is False
    assert len(result.unsupported_claims) == 1
    assert "Google" in result.unsupported_claims[0]


def test_grounding_uses_strong_lexical_fallback_when_nli_is_neutral():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a probabilistic lexical ranking function that "
            "considers query-term frequency, document length, and "
            "the rarity of terms across the collection."
        )
    ]

    verifier._predict_nli = (
        lambda premise, hypothesis: ("neutral", 0.99)
    )

    result = verifier.verify_grounding(
        "BM25 is a probabilistic lexical ranking function that "
        "considers query-term frequency and document length. "
        "[Evidence 1]",
        evidence,
    )

    assert result.supported is True
    assert result.cited_evidence == [1]
    assert result.unsupported_claims == []
    assert "lexical" in result.reason.lower()


def test_grounding_rejects_weak_lexical_overlap():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a probabilistic lexical ranking function that "
            "considers query-term frequency, document length, and "
            "the rarity of terms across the collection."
        )
    ]

    verifier._predict_nli = (
        lambda premise, hypothesis: ("neutral", 0.99)
    )

    result = verifier.verify_grounding(
        "BM25 supports image segmentation in medical imaging. "
        "[Evidence 1]",
        evidence,
    )

    assert result.supported is False
    assert len(result.unsupported_claims) == 1


def test_grounding_supports_multiple_cited_evidence_blocks():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a lexical retrieval method based on term frequency."
        ),
        make_evidence(
            "BM25 also accounts for document length when calculating relevance."
        ),
    ]

    verifier._predict_nli = (
        lambda premise, hypothesis: (
            "entailment",
            0.95,
        )
    )

    result = verifier.verify_grounding(
        "BM25 uses term frequency and document length. "
        "[Evidence 1] [Evidence 2]",
        evidence,
    )

    assert result.supported is True
    assert result.cited_evidence == [1, 2]
    assert result.unsupported_claims == []


def test_grounding_extracts_citation_attached_to_previous_sentence():
    verifier = EvidenceVerifier()

    evidence = [
        make_evidence(
            "BM25 is a probabilistic lexical ranking function "
            "used for information retrieval."
        )
    ]

    verifier._predict_nli = (
        lambda premise, hypothesis: (
            "entailment",
            0.95,
        )
    )

    result = verifier.verify_grounding(
        "BM25 is a probabilistic lexical ranking function. "
        "[Evidence 1]",
        evidence,
    )

    assert result.supported is True
    assert result.cited_evidence == [1]
    assert result.unsupported_claims == []