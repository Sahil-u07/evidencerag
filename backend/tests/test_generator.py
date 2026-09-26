from types import SimpleNamespace

from app.generation.generator import OpenAIGenerator


class FakeResponses:
    def create(self, **kwargs):
        self.kwargs = kwargs

        return SimpleNamespace(
            output_text=(
                "RRF combines ranked lists. [Evidence 1]"
            )
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def make_evidence(
    text="RRF combines multiple ranked lists.",
    source="information_retrieval.md",
    chunk_id="ir:1",
    score=0.9,
):
    chunk = SimpleNamespace(
        text=text,
        source=source,
        page=None,
        chunk_id=chunk_id,
    )

    search_result = SimpleNamespace(
        chunk=chunk,
        score=score,
    )

    # Match the real RerankedResult structure:
    #
    # RerankedResult
    # ├── result
    # │   └── chunk
    # └── score
    return SimpleNamespace(
        result=search_result,
        score=score,
    )


def test_generator_uses_retrieved_evidence():
    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.client = FakeClient()
    generator.model = "test-model"

    evidence = [
        make_evidence(
            "Reciprocal Rank Fusion combines multiple ranked lists."
        )
    ]

    result = generator.generate(
        "What is Reciprocal Rank Fusion?",
        evidence,
    )

    assert "RRF combines ranked lists" in result.answer
    assert "[Evidence 1]" in result.answer

    request = generator.client.responses.kwargs

    assert request["model"] == "test-model"
    assert (
        "Reciprocal Rank Fusion"
        in request["input"]
    )


def test_generator_rejects_empty_query():
    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.client = FakeClient()
    generator.model = "test-model"

    try:
        generator.generate("", [])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_generator_abstains_without_evidence():
    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.client = FakeClient()
    generator.model = "test-model"

    result = generator.generate(
        "What is something not present in the documents?",
        [],
    )

    assert (
        "don't have enough evidence"
        in result.answer
    )