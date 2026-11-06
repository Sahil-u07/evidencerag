from types import SimpleNamespace

import pytest

from app.generation.ollama_generator import GeneratedAnswer
from app.generation.pipeline import (
    ABSTENTION_MESSAGE,
    RAGPipeline,
)
from app.generation.verifier import EvidenceVerifier
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    def search(self, query, top_k=5):
        chunk = SimpleNamespace(
            text="RRF combines ranked retrieval results.",
            source="information_retrieval.md",
            page=None,
            chunk_id="ir:5",
        )

        search_result = SimpleNamespace(
            chunk=chunk,
            score=0.95,
        )

        return [
            SimpleNamespace(
                result=search_result,
                score=0.95,
            )
        ]


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.received_query = None
        self.received_evidence = None

    def generate(self, query, evidence):
        self.received_query = query
        self.received_evidence = evidence

        return GeneratedAnswer(answer=self.answer)


def test_pipeline_returns_verified_answer():
    retriever = FakeRetriever()

    generator = FakeGenerator(
        "RRF combines ranked retrieval results. [Evidence 1]"
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.ask("What is RRF?")

    assert result.query == "What is RRF?"
    assert len(result.evidence) == 1
    assert "RRF combines" in result.answer.answer

    assert generator.received_query == "What is RRF?"
    assert generator.received_evidence == result.evidence

    assert result.verification.supported is True
    assert result.verification.cited_evidence == [1]
    assert result.answer.answer != ABSTENTION_MESSAGE


def test_pipeline_records_metrics():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(
            "RRF combines ranked retrieval results. [Evidence 1]"
        ),
    )

    result = pipeline.ask(
        "What is RRF?",
        top_k=5,
    )

    assert result.metrics.query == "What is RRF?"
    assert result.metrics.top_k == 5
    assert result.metrics.retrieved_evidence_count == 1
    assert result.metrics.cited_evidence_count == 1
    assert result.metrics.verification_supported is True
    assert result.metrics.latency_ms >= 0


def test_pipeline_records_abstention_metrics():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(
            "Quantum computers use superconducting qubits. "
            "[Evidence 1]"
        ),
    )

    result = pipeline.ask("What is RRF?")

    assert result.verification.supported is False
    assert result.answer.answer == ABSTENTION_MESSAGE

    assert result.metrics.query == "What is RRF?"
    assert result.metrics.top_k == 5
    assert result.metrics.retrieved_evidence_count == 1
    assert result.metrics.cited_evidence_count == 1
    assert result.metrics.verification_supported is False
    assert result.metrics.latency_ms >= 0


def test_pipeline_abstains_when_answer_is_not_grounded():
    retriever = FakeRetriever()

    generator = FakeGenerator(
        "Quantum computers use superconducting qubits. "
        "[Evidence 1]"
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.ask("What is RRF?")

    assert result.verification.supported is False
    assert result.verification.cited_evidence == [1]
    assert len(result.verification.unsupported_claims) == 1
    assert result.answer.answer == ABSTENTION_MESSAGE


def test_pipeline_rejects_empty_query():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(
            "RRF combines ranked retrieval results. [Evidence 1]"
        ),
    )

    try:
        pipeline.ask("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_empty_index_abstains_through_real_components():
    class EmbedderStub:
        def encode_query(self, query):
            return [1.0, 0.0]

    retriever = RerankedHybridRetriever(
        hybrid_retriever=HybridRetriever(
            embedder=EmbedderStub(),
            dense_retriever=DenseRetriever(),
            bm25_retriever=BM25Retriever(),
        ),
        # __new__ skips loading model weights; empty input never uses them
        reranker=CrossEncoderReranker.__new__(CrossEncoderReranker),
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=FakeGenerator("RRF combines ranked lists. [Evidence 1]"),
        verifier=EvidenceVerifier(),
    )

    result = pipeline.ask("What is RRF?")

    assert result.evidence == []
    assert result.verification.supported is False
    assert "No evidence was retrieved" in result.verification.reason
    assert result.answer.answer == ABSTENTION_MESSAGE


def test_attach_citation_keeps_bullets_on_separate_lines():
    pipeline = RAGPipeline(retriever=FakeRetriever(), generator=FakeGenerator(""), verifier=EvidenceVerifier())
    evidence = [
        SimpleNamespace(result=SimpleNamespace(chunk=SimpleNamespace(text=text)))
        for text in (
            "TCP uses a three way handshake to open connections.",
            "UDP is connectionless and unreliable.",
        )
    ]

    cited = pipeline.attach_citation(
        "## Overview\n- TCP uses a three way handshake\n\n- UDP is connectionless and unreliable",
        evidence,
    )

    assert cited == (
        "## Overview\n"
        "- TCP uses a three way handshake [Evidence 1]\n"
        "- UDP is connectionless and unreliable [Evidence 2]"
    )
    assert pipeline.verifier._extract_claims(cited) == [
        ("TCP uses a three way handshake", [1]),
        ("UDP is connectionless and unreliable", [2]),
    ]


def test_ask_checks_only_the_passages_the_model_saw_and_trims_a_cut_off_answer():
    class FiveResults(FakeRetriever):
        def search(self, query, top_k=5):
            return super().search(query, top_k) * 5

    calls = []

    class Verifier:
        def verify_grounding(self, answer, evidence):
            calls.append((answer, len(evidence)))
            return SimpleNamespace(supported=True, cited_evidence=[1], unsupported_claims=[], reason="")

    generator = FakeGenerator("")
    generator.generate = lambda query, evidence: GeneratedAnswer(
        answer="RRF combines ranked retrieval results. It then so", truncated=True
    )

    result = RAGPipeline(retriever=FiveResults(), generator=generator, verifier=Verifier()).ask("What is RRF?")

    assert calls == [("RRF combines ranked retrieval results. [Evidence 1]", 3)]
    assert result.answer.answer == "RRF combines ranked retrieval results. [Evidence 1]"
    assert len(result.evidence) == 5


def test_pipeline_requires_a_generator():
    with pytest.raises(TypeError):
        RAGPipeline(retriever=FakeRetriever())
