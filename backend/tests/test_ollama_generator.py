from types import SimpleNamespace

from app.generation.ollama_generator import OllamaGenerator


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.text}


def make_evidence(
    text="RRF combines ranked retrieval results.",
    source="information_retrieval.md",
    chunk_id="information_retrieval.md:5",
    score=5.5,
):
    chunk = SimpleNamespace(
        text=text,
        source=source,
        page=None,
        chunk_id=chunk_id,
    )

    search_result = SimpleNamespace(
        chunk=chunk,
        score=0.9,
    )

    return SimpleNamespace(
        result=search_result,
        score=score,
    )


def test_ollama_generator_uses_reranked_evidence(monkeypatch):
    captured_request = {}

    def fake_post(url, json, timeout):
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout

        return FakeResponse(
            "RRF combines ranked retrieval results. [Evidence 1]"
        )

    monkeypatch.setattr(
        "app.generation.ollama_generator.requests.post",
        fake_post,
    )

    generator = OllamaGenerator()

    evidence = [
        make_evidence()
    ]

    result = generator.generate(
        "What is RRF?",
        evidence,
    )

    assert "RRF combines" in result.answer
    assert "[Evidence 1]" in result.answer

    assert captured_request["json"]["model"] == "llama3.2:3b"
    assert "What is RRF?" in captured_request["json"]["prompt"]
    assert "RRF combines ranked retrieval results." in (
        captured_request["json"]["prompt"]
    )


def test_ollama_generator_rejects_empty_query():
    generator = OllamaGenerator()

    try:
        generator.generate("", [])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_ollama_generator_abstains_without_evidence():
    generator = OllamaGenerator()

    result = generator.generate(
        "What is something not in the documents?",
        [],
    )

    assert "don't have enough evidence" in result.answer