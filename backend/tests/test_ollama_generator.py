from types import SimpleNamespace
import json

from app.generation.ollama_generator import OllamaGenerator


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "response": self.payload
        }


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


def make_json_response(
    answer="RRF combines ranked retrieval results. [Evidence 1]",
    citations=None,
):
    if citations is None:
        citations = [1]

    return json.dumps(
        {
            "answer": answer,
            "citations": citations,
        }
    )


def test_ollama_generator_uses_reranked_evidence(monkeypatch):
    captured_request = {}

    def fake_post(url, json, timeout):
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout

        return FakeResponse(
            make_json_response(
                answer=(
                    "RRF combines ranked retrieval results. "
                    "[Evidence 1]"
                ),
                citations=[1],
            )
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

    assert captured_request["json"]["model"] == (
        "llama3.2:3b"
    )

    assert captured_request["json"]["format"] == "json"

    assert captured_request["json"]["stream"] is False

    assert (
        captured_request["json"]["options"]["temperature"]
        == 0.1
    )

    assert "What is RRF?" in (
        captured_request["json"]["prompt"]
    )

    assert (
        "RRF combines ranked retrieval results."
        in captured_request["json"]["prompt"]
    )


def test_ollama_generator_rejects_empty_query():
    generator = OllamaGenerator()

    try:
        generator.generate(
            "",
            [],
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty"


def test_ollama_generator_abstains_without_evidence():
    generator = OllamaGenerator()

    result = generator.generate(
        "What is something not in the documents?",
        [],
    )

    assert (
        "don't have enough evidence"
        in result.answer
    )


def test_ollama_generator_rejects_invalid_json(
    monkeypatch,
):
    def fake_post(url, json, timeout):
        return FakeResponse(
            "This is not valid JSON."
        )

    monkeypatch.setattr(
        "app.generation.ollama_generator.requests.post",
        fake_post,
    )

    generator = OllamaGenerator()

    try:
        generator.generate(
            "What is RRF?",
            [make_evidence()],
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == (
            "Ollama returned invalid JSON."
        )


def test_ollama_generator_rejects_invalid_citation(
    monkeypatch,
):
    def fake_post(url, json, timeout):
        return FakeResponse(
            make_json_response(
                answer=(
                    "RRF combines ranked retrieval results. "
                    "[Evidence 3]"
                ),
                citations=[3],
            )
        )

    monkeypatch.setattr(
        "app.generation.ollama_generator.requests.post",
        fake_post,
    )

    generator = OllamaGenerator()

    try:
        generator.generate(
            "What is RRF?",
            [make_evidence()],
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert (
            "invalid evidence citation"
            in str(exc)
        )


def test_ollama_generator_adds_missing_inline_citation(
    monkeypatch,
):
    def fake_post(url, json, timeout):
        return FakeResponse(
            make_json_response(
                answer=(
                    "RRF combines ranked retrieval results."
                ),
                citations=[1],
            )
        )

    monkeypatch.setattr(
        "app.generation.ollama_generator.requests.post",
        fake_post,
    )

    generator = OllamaGenerator()

    result = generator.generate(
        "What is RRF?",
        [make_evidence()],
    )

    assert result.answer == (
        "RRF combines ranked retrieval results. "
        "[Evidence 1]"
    )