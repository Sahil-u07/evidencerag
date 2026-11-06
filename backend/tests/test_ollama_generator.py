import json

import pytest

from app.core.config import settings
from app.generation import streaming
from app.generation.ollama_generator import OllamaGenerator
from app.generation.streaming import ABSTENTION_MESSAGE, GENERATION_CONTEXT_K


class FakeResponse:
    def __init__(self, tokens, done_reason="stop"):
        self.lines = [
            json.dumps({"response": token, "done": False}) for token in tokens
        ]
        self.lines.append(
            json.dumps({"response": "", "done": True, "done_reason": done_reason})
        )

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def close(self):
        pass


def make_evidence(text="RRF combines ranked retrieval results."):
    class Chunk:
        page = 5
        source = "retrieval.md"
        chunk_id = "chunk-1"

    Chunk.text = text

    class Result:
        chunk = Chunk()

    class Evidence:
        result = Result()

    return Evidence()


def test_generate_joins_the_stream_into_one_answer(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResponse(["RRF combines ", "ranked results. "])

    monkeypatch.setattr(streaming.requests, "post", fake_post)

    result = OllamaGenerator().generate("What is RRF?", [make_evidence()])

    assert result.answer == "RRF combines ranked results."
    assert result.truncated is False
    assert "What is RRF?" in captured["payload"]["prompt"]
    assert "RRF combines ranked retrieval results." in captured["payload"]["prompt"]
    assert captured["payload"]["stream"] is True
    assert "format" not in captured["payload"]


def test_generate_shows_the_model_only_the_top_passages(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured["prompt"] = kwargs["json"]["prompt"]
        return FakeResponse(["Answer."])

    monkeypatch.setattr(streaming.requests, "post", fake_post)
    evidence = [
        make_evidence(f"Passage number {number}.")
        for number in range(GENERATION_CONTEXT_K + 1)
    ]

    OllamaGenerator().generate("What is RRF?", evidence)

    assert f"Passage number {GENERATION_CONTEXT_K - 1}." in captured["prompt"]
    assert f"Passage number {GENERATION_CONTEXT_K}." not in captured["prompt"]


def test_generate_reports_a_length_cutoff(monkeypatch):
    monkeypatch.setattr(
        streaming.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(["Cut of"], "length"),
    )

    assert (
        OllamaGenerator().generate("What is RRF?", [make_evidence()]).truncated is True
    )


def test_generate_rejects_empty_query():
    with pytest.raises(ValueError):
        OllamaGenerator().generate("", [make_evidence()])


def test_generate_abstains_with_no_evidence():
    assert OllamaGenerator().generate("What is RRF?", []).answer == ABSTENTION_MESSAGE


def test_generate_uses_the_configured_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "ollama_url", "http://ollama.test:1234")
    monkeypatch.setattr(settings, "ollama_model", "tiny:1b")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["model"] = kwargs["json"]["model"]
        return FakeResponse(["Answer."])

    monkeypatch.setattr(streaming.requests, "post", fake_post)

    OllamaGenerator().generate("What is RRF?", [make_evidence()])

    assert captured == {
        "url": "http://ollama.test:1234/api/generate",
        "model": "tiny:1b",
    }
