import json

import pytest

from app.generation.ollama_generator import OllamaGenerator


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.payload)


def make_evidence():
    class Chunk:
        text = "RRF combines ranked retrieval results."
        page = 5
        source = "retrieval.md"
        chunk_id = "chunk-1"

    class Result:
        chunk = Chunk()

    class Evidence:
        result = Result()

    return Evidence()


def make_ollama_response(answer):
    return FakeResponse(
        json.dumps(
            {
                "response": json.dumps(
                    {
                        "answer": answer,
                    }
                )
            }
        )
    )


def test_ollama_generator_uses_reranked_evidence(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured["payload"] = kwargs["json"]

        return make_ollama_response(
            "RRF combines ranked retrieval results."
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

    assert result.answer == "RRF combines ranked retrieval results."
    assert "[Evidence 1]" not in result.answer
    assert "What is RRF?" in captured["payload"]["prompt"]
    assert (
        "RRF combines ranked retrieval results."
        in captured["payload"]["prompt"]
    )
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0.1


def test_ollama_generator_rejects_empty_query():
    generator = OllamaGenerator()

    with pytest.raises(ValueError):
        generator.generate("", [make_evidence()])


def test_ollama_generator_handles_no_evidence():
    generator = OllamaGenerator()

    result = generator.generate(
        "What is RRF?",
        [],
    )

    assert "enough evidence" in result.answer.lower()


def test_ollama_generator_rejects_invalid_json(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "response": "not valid json",
                }
            )
        )

    monkeypatch.setattr(
        "app.generation.ollama_generator.requests.post",
        fake_post,
    )

    generator = OllamaGenerator()

    with pytest.raises(RuntimeError):
        generator.generate(
            "What is RRF?",
            [make_evidence()],
        )


def test_ollama_generator_returns_plain_grounded_answer(monkeypatch):
    def fake_post(*args, **kwargs):
        return make_ollama_response(
            "RRF combines ranked retrieval results."
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

    assert result.answer == "RRF combines ranked retrieval results."


def test_ollama_generator_does_not_delegate_citations(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "response": json.dumps(
                        {
                            "answer": (
                                "RRF combines ranked retrieval results."
                            ),
                            "citations": [1],
                        }
                    )
                }
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
        "RRF combines ranked retrieval results."
    )
    assert "[Evidence 1]" not in result.answer
