from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeRetriever:
    def __init__(self):
        self.chunk = SimpleNamespace(
            text="RRF combines ranked retrieval results.",
            source="information_retrieval.md",
            page=None,
            chunk_id="information_retrieval.md:5",
        )

    def search(self, query, top_k=5):
        return [
            SimpleNamespace(
                result=SimpleNamespace(
                    chunk=self.chunk,
                ),
                score=5.5,
            )
        ]


class FakePipeline:
    def ask(self, query, top_k=5):
        evidence = [
            SimpleNamespace(
                result=SimpleNamespace(
                    chunk=SimpleNamespace(
                        text="RRF combines ranked retrieval results.",
                        source="information_retrieval.md",
                        page=None,
                        chunk_id="information_retrieval.md:5",
                    )
                ),
                score=5.5,
            )
        ]

        verification = SimpleNamespace(
            supported=True,
            cited_evidence=[1],
            unsupported_claims=[],
            reason="All cited claims have sufficient textual support.",
        )

        return SimpleNamespace(
            query=query,
            answer=SimpleNamespace(
                answer=(
                    "RRF combines ranked retrieval results. "
                    "[Evidence 1]"
                )
            ),
            evidence=evidence,
            verification=verification,
        )


@pytest.fixture
def fake_retriever():
    return FakeRetriever()


@pytest.fixture
def client(monkeypatch, tmp_path, fake_retriever):
    monkeypatch.setattr(
        main,
        "build_retriever",
        lambda: fake_retriever,
    )

    monkeypatch.setattr(
        main,
        "UPLOAD_DIR",
        tmp_path,
    )

    with TestClient(main.app) as test_client:
        yield test_client

    main.state.pipeline = None
    main.state.retriever = None


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["generator"] == "ollama"
    assert data["model"] == "llama3.2:3b"


def test_search_endpoint(client):
    response = client.post(
        "/search",
        json={
            "query": "What is reciprocal rank fusion?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is reciprocal rank fusion?"
    assert len(data["results"]) == 1

    result = data["results"][0]

    assert result["evidence_id"] == 1
    assert result["source"] == "information_retrieval.md"
    assert result["chunk_id"] == "information_retrieval.md:5"
    assert result["text"] == (
        "RRF combines ranked retrieval results."
    )
    assert result["score"] == 5.5


def test_ask_endpoint(monkeypatch, client):
    main.state.pipeline = FakePipeline()

    response = client.post(
        "/ask",
        json={
            "query": "What is reciprocal rank fusion?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == (
        "What is reciprocal rank fusion?"
    )

    assert data["answer"] == (
        "RRF combines ranked retrieval results. "
        "[Evidence 1]"
    )

    assert data["verification"]["supported"] is True
    assert data["verification"]["cited_evidence"] == [1]
    assert data["verification"]["unsupported_claims"] == []

    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["evidence_id"] == 1


def test_search_rejects_empty_query(client):
    response = client.post(
        "/search",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_ask_rejects_empty_query(client):
    response = client.post(
        "/ask",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_search_rejects_invalid_top_k(client):
    response = client.post(
        "/search",
        json={
            "query": "RRF",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/ask",
        json={
            "query": "RRF",
            "top_k": 21,
        },
    )

    assert response.status_code == 422


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "malicious.exe",
                b"not a supported document",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "Unsupported file type" in data["detail"]


def test_upload_rejects_oversized_file(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        main.settings,
        "max_upload_size_mb",
        1,
    )

    oversized_content = b"x" * (
        1 * 1024 * 1024 + 1
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large_document.md",
                oversized_content,
                "text/markdown",
            )
        },
    )

    assert response.status_code == 413

    data = response.json()

    assert "File is too large" in data["detail"]
    assert "1 MB" in data["detail"]


def test_upload_document(client, tmp_path):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "test_document.md",
                b"# Test Document\n\nRRF combines ranked lists.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == (
        "Document uploaded and indexed successfully."
    )
    assert data["filename"] == "test_document.md"
    assert data["stored_as"].endswith(
        "_test_document.md"
    )

    uploaded_files = list(tmp_path.iterdir())

    assert len(uploaded_files) == 1
    assert uploaded_files[0].read_text(
        encoding="utf-8"
    ) == "# Test Document\n\nRRF combines ranked lists."