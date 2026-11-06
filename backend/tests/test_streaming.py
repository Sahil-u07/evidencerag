import json
from types import SimpleNamespace

import pytest
import requests

import app.main as main
from app.core.config import settings
from app.generation import pipeline as pipeline_module
from app.generation import streaming
from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.generation.streaming import (
    ABSTENTION_MESSAGE,
    StreamToken,
    build_prompt,
    stream_generate,
    trim_unfinished_sentence,
)


def make_evidence(text, number):
    chunk = SimpleNamespace(text=text, source="doc.md", page=None, chunk_id=f"doc.md:{number}")
    return SimpleNamespace(result=SimpleNamespace(chunk=chunk), score=1.0)


class RecordingVerifier:
    def __init__(self):
        self.calls = []

    def verify_grounding(self, answer, evidence):
        self.calls.append((answer, len(evidence)))
        return SimpleNamespace(supported=False, cited_evidence=[], unsupported_claims=[], reason="x")


def run_stream(monkeypatch, evidence, tokens):
    verifier = RecordingVerifier()
    retriever = SimpleNamespace(search=lambda query, top_k: evidence)
    monkeypatch.setattr(
        main.state, "pipeline", RAGPipeline(retriever=retriever, generator=object(), verifier=verifier)
    )
    monkeypatch.setattr(main, "stream_generate", lambda query, evidence: iter(tokens))

    events = []
    for raw in main.stream_query_events("question", top_k=5):
        name, data = raw.strip().split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))

    return events, verifier


def test_stream_verifies_only_the_passages_the_model_saw(monkeypatch):
    texts = [f"Unrelated filler passage number {word}." for word in ("alpha", "bravo", "charlie")]
    texts += ["The handshake finishes after the server sends the finished message."] * 2
    evidence = [make_evidence(text, number) for number, text in enumerate(texts, start=1)]

    events, verifier = run_stream(
        monkeypatch,
        evidence,
        [StreamToken("The handshake finishes after the server sends the finished message.")],
    )

    assert verifier.calls == [("The handshake finishes after the server sends the finished message.", 3)]
    assert len(dict((name, data) for name, data in events)["sources"]["items"]) == 5


def test_cut_off_last_sentence_is_not_verified(monkeypatch):
    evidence = [make_evidence("The server sends a reset segment to the client.", 1)]

    events, verifier = run_stream(
        monkeypatch,
        evidence,
        [
            StreamToken("The server sends a reset segment to the client. The client then clo"),
            StreamToken("", truncated=True),
        ],
    )

    assert verifier.calls == [("The server sends a reset segment to the client. [Evidence 1]", 1)]
    assert [data["t"] for name, data in events if name == "token"] == [
        "The server sends a reset segment to the client. The client then clo"
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two", "One."),
        ("One.\nTwo is cut", "One."),
        ("One. Two.", "One. Two."),
        ("One is cut", "One is cut"),
        ("", ""),
    ],
)
def test_trim_unfinished_sentence(text, expected):
    assert trim_unfinished_sentence(text) == expected


class FakeResponse:
    def __init__(self, done_reason):
        self.lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": "", "done": True, "done_reason": done_reason}),
        ]

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def close(self):
        pass


@pytest.mark.parametrize(("done_reason", "truncated"), [("length", True), ("stop", False)])
def test_stream_reports_a_length_cutoff(monkeypatch, done_reason, truncated):
    monkeypatch.setattr(streaming.requests, "post", lambda *args, **kwargs: FakeResponse(done_reason))

    tokens = list(stream_generate("question", [make_evidence("text", 1)]))

    assert [token.text for token in tokens if token.text] == ["Hello"]
    assert any(token.truncated for token in tokens) is truncated


def test_stream_uses_the_configured_endpoint(monkeypatch):
    monkeypatch.setattr(streaming.settings, "ollama_url", "http://ollama.test:1234")
    monkeypatch.setattr(streaming.settings, "ollama_model", "tiny:1b")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["model"] = kwargs["json"]["model"]
        return FakeResponse("stop")

    monkeypatch.setattr(streaming.requests, "post", fake_post)

    list(stream_generate("question", [make_evidence("text", 1)]))

    assert captured == {"url": "http://ollama.test:1234/api/generate", "model": "tiny:1b"}


def test_a_stream_keeps_the_pipeline_it_started_with(monkeypatch):
    retriever = SimpleNamespace(search=lambda query, top_k: [make_evidence("text", 1)])
    old_verifier, new_verifier = RecordingVerifier(), RecordingVerifier()
    old = RAGPipeline(retriever=retriever, generator=object(), verifier=old_verifier)
    new = RAGPipeline(retriever=retriever, generator=object(), verifier=new_verifier)

    def swap_during_generation(query, evidence):
        main.state.pipeline = new
        yield StreamToken("Some answer.")

    monkeypatch.setattr(main.state, "pipeline", old)
    monkeypatch.setattr(main, "stream_generate", swap_during_generation)

    list(main.stream_query_events("question", top_k=5))

    assert len(old_verifier.calls) == 1
    assert new_verifier.calls == []


def test_every_path_uses_the_one_abstention_message(monkeypatch):
    events, _ = run_stream(monkeypatch, [make_evidence("text", 1)], [StreamToken("Some answer.")])

    assert dict(events)["final"]["answer"] == ABSTENTION_MESSAGE
    assert OllamaGenerator().generate("question", []).answer == ABSTENTION_MESSAGE
    assert pipeline_module.ABSTENTION_MESSAGE is ABSTENTION_MESSAGE
    assert ABSTENTION_MESSAGE in build_prompt("question", [make_evidence("text", 1)])


def test_verification_event_has_the_fields_the_frontend_reads(monkeypatch):
    events, _ = run_stream(monkeypatch, [make_evidence("text", 1)], [StreamToken("Some answer.")])

    verification = dict(events)["verification"]

    assert verification == {
        "supported": False,
        "cited_evidence": [],
        "unsupported_claims": [],
        "reason": "x",
    }


def run_failing_stream(monkeypatch, error):
    def fail(query, evidence):
        raise error

    retriever = SimpleNamespace(search=lambda query, top_k: [])
    monkeypatch.setattr(main.state, "pipeline", SimpleNamespace(retriever=retriever))
    monkeypatch.setattr(main, "stream_generate", fail)

    events = []
    for raw in main.stream_query_events("question", top_k=5):
        name, data = raw.strip().split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))

    return events


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            requests.ConnectionError("HTTPConnectionPool(host='localhost') refused"),
            f"Could not reach Ollama at {settings.ollama_url.rstrip('/')}. Start Ollama and try again.",
        ),
        (
            requests.Timeout("read timed out"),
            "Ollama did not respond in time. It may still be loading the model; try again.",
        ),
        (
            requests.HTTPError("404 for url http://x/api/generate", response=SimpleNamespace(status_code=404)),
            f"The model {settings.ollama_model} is not available in Ollama. "
            f"Run: ollama pull {settings.ollama_model}",
        ),
        (
            requests.HTTPError("500 for url http://x/api/generate", response=SimpleNamespace(status_code=500)),
            "Something went wrong while answering. Check the backend log.",
        ),
        (
            RuntimeError("secret internal detail"),
            "Something went wrong while answering. Check the backend log.",
        ),
    ],
)
def test_stream_errors_are_plain_and_do_not_leak(monkeypatch, caplog, error, message):
    events = run_failing_stream(monkeypatch, error)

    assert events[-1] == ("error", {"message": message})
    assert any(record.exc_info for record in caplog.records)
