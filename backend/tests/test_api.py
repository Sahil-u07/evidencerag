from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

import app.main as main
from app.core.config import settings


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

        metrics = SimpleNamespace(
            query=query,
            top_k=top_k,
            retrieved_evidence_count=1,
            cited_evidence_count=1,
            verification_supported=True,
            latency_ms=12.34,
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
            metrics=metrics,
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

    def ollama_down(url, timeout):
        raise requests.ConnectionError("no network in tests")

    monkeypatch.setattr(main.requests, "get", ollama_down)

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
    assert data["model"] == settings.ollama_model


def test_request_id_header(client):
    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert len(request_id) == 36


def test_request_ids_are_unique(client):
    first_response = client.get("/health")
    second_response = client.get("/health")

    first_request_id = first_response.headers.get(
        "X-Request-ID"
    )
    second_request_id = second_response.headers.get(
        "X-Request-ID"
    )

    assert first_request_id is not None
    assert second_request_id is not None
    assert first_request_id != second_request_id


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

    assert data["metrics"]["query"] == (
        "What is reciprocal rank fusion?"
    )
    assert data["metrics"]["top_k"] == 5
    assert data["metrics"]["retrieved_evidence_count"] == 1
    assert data["metrics"]["cited_evidence_count"] == 1
    assert data["metrics"]["verification_supported"] is True
    assert data["metrics"]["latency_ms"] == 12.34


def test_list_documents_empty(client):
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["documents"] == []


def test_list_documents(client, tmp_path):
    document = tmp_path / "test_document.md"
    document.write_text(
        "# Test Document\n\nRRF combines ranked lists.",
        encoding="utf-8",
    )

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert len(data["documents"]) == 1

    item = data["documents"][0]

    assert item["filename"] == "test_document.md"
    assert item["extension"] == ".md"
    assert item["size_bytes"] == document.stat().st_size


def test_delete_document(client, tmp_path, monkeypatch):
    document = tmp_path / "test_document.md"
    document.write_text(
        "# Test Document\n\nRRF combines ranked lists.",
        encoding="utf-8",
    )

    rebuild_calls = []

    monkeypatch.setattr(
        main,
        "rebuild_pipeline",
        lambda: rebuild_calls.append(True),
    )

    response = client.delete(
        "/documents/test_document.md"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == (
        "Document deleted and index rebuilt successfully."
    )
    assert data["filename"] == "test_document.md"

    assert not document.exists()
    assert rebuild_calls == [True]


def test_delete_document_not_found(client):
    response = client.delete(
        "/documents/missing_document.md"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found."


def test_delete_document_rejects_path_traversal(client):
    response = client.delete(
        "/documents/../secret.md"
    )

    assert response.status_code in {400, 404}


def test_delete_document_rejects_unsupported_file(client):
    response = client.delete(
        "/documents/malicious.exe"
    )

    assert response.status_code == 400

    data = response.json()

    assert "Unsupported file type" in data["detail"]


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


def test_upload_rejects_a_file_with_no_text(client, tmp_path):
    response = client.post(
        "/documents/upload",
        files={"file": ("blank.txt", b"   \n  ", "text/plain")},
    )

    assert response.status_code == 400
    assert "No text could be extracted" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def fake_tags(models):
    def get(url, timeout):
        assert url == f"{settings.ollama_url.rstrip('/')}/api/tags"
        assert timeout == 2
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"models": [{"name": name} for name in models]},
        )

    return get


def test_health_reports_the_settings(client, monkeypatch):
    monkeypatch.setattr(main.requests, "get", fake_tags([settings.ollama_model]))

    assert client.get("/health").json() == {
        "status": "ok",
        "ready": True,
        "generator": "ollama",
        "model": settings.ollama_model,
        "generator_available": True,
        "max_upload_mb": settings.max_upload_size_mb,
    }


def test_health_generator_unavailable_when_ollama_is_down(client, monkeypatch):
    def down(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(main.requests, "get", down)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["generator_available"] is False


def test_health_generator_unavailable_when_the_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(main.requests, "get", fake_tags(["other:7b"]))

    assert client.get("/health").json()["generator_available"] is False


def test_health_accepts_an_untagged_model_name(client, monkeypatch):
    monkeypatch.setattr(settings, "ollama_model", "tiny")
    monkeypatch.setattr(main.requests, "get", fake_tags(["tiny:latest"]))

    assert client.get("/health").json()["generator_available"] is True


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            requests.ConnectionError("refused"),
            f"Could not reach Ollama at {settings.ollama_url.rstrip('/')}. Start Ollama and try again.",
        ),
        (
            requests.Timeout("read timed out"),
            "Ollama did not respond in time. It may still be loading the model; try again.",
        ),
        (RuntimeError("secret internal detail"), "Something went wrong while answering. Check the backend log."),
    ],
)
def test_ask_errors_are_plain_and_do_not_leak(client, error, message):
    def fail(query, top_k=5):
        raise error

    main.state.pipeline = SimpleNamespace(ask=fail)

    response = client.post("/ask", json={"query": "What is RRF?"})

    assert response.status_code == 503
    assert response.json()["detail"] == message


def test_missing_upload_folder_is_created(client, monkeypatch, tmp_path):
    missing = tmp_path / "data" / "uploads"
    monkeypatch.setattr(main, "UPLOAD_DIR", missing)

    assert client.get("/documents").json() == {"documents": []}
    assert missing.is_dir()


def test_upload_dir_comes_from_the_setting():
    assert main.UPLOAD_DIR == main.PROJECT_ROOT / settings.upload_dir


def test_error_message_has_no_double_slash(client, monkeypatch):
    monkeypatch.setattr(settings, "ollama_url", "http://ollama.test:1234/")

    def fail(query, top_k=5):
        raise requests.ConnectionError("refused")

    main.state.pipeline = SimpleNamespace(ask=fail)

    detail = client.post("/ask", json={"query": "What is RRF?"}).json()["detail"]

    assert detail == "Could not reach Ollama at http://ollama.test:1234. Start Ollama and try again."
