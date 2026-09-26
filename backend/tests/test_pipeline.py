from types import SimpleNamespace

from app.generation.pipeline import RAGPipeline


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
    def __init__(self):
        self.received_query = None
        self.received_evidence = None

    def generate(self, query, evidence):
        self.received_query = query
        self.received_evidence = evidence

        return SimpleNamespace(
            answer="RRF combines ranked retrieval results. [Evidence 1]"
        )


def test_pipeline_connects_retrieval_and_generation():
    retriever = FakeRetriever()
    generator = FakeGenerator()

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


def test_pipeline_rejects_empty_query():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    try:
        pipeline.ask("")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"