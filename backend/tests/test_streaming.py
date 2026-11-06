import json
from types import SimpleNamespace

import pytest

import app.main as main
from app.generation import streaming
from app.generation.pipeline import RAGPipeline
from app.generation.streaming import StreamToken, stream_generate, trim_unfinished_sentence


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
    monkeypatch.setattr(main.state, "retriever", SimpleNamespace(search=lambda query, top_k: evidence))
    monkeypatch.setattr(
        main.state, "pipeline", RAGPipeline(retriever=None, generator=object(), verifier=verifier)
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
