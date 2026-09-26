from types import SimpleNamespace

from app.generation.pipeline import (
    ABSTENTION_MESSAGE,
    RAGPipeline,
)


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

        return SimpleNamespace(
            answer=self.answer
        )


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