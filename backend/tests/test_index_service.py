from pathlib import Path

from app.retrieval.dense import DenseRetriever
from app.retrieval.index_service import IndexService


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    def encode_query(self, query: str) -> list[float]:
        return (
            [1.0, 0.0]
            if "python" in query.lower()
            else [0.0, 1.0]
        )


def test_index_service_indexes_text_document(tmp_path: Path):
    file_path = tmp_path / "notes.txt"

    file_path.write_text(
        "Python is used for machine learning. "
        "Cooking requires ingredients.",
        encoding="utf-8",
    )

    service = IndexService(
        embedder=FakeEmbedder(),
        retriever=DenseRetriever(),
    )

    chunk_count = service.index_document(
        file_path,
        chunk_size=20,
        overlap=5,
    )

    assert chunk_count > 0


def test_index_service_returns_relevant_result(tmp_path: Path):
    file_path = tmp_path / "notes.txt"

    file_path.write_text(
        "Python is widely used in machine learning. "
        "Cooking involves recipes and ingredients.",
        encoding="utf-8",
    )

    service = IndexService(
        embedder=FakeEmbedder(),
        retriever=DenseRetriever(),
    )

    service.index_document(
        file_path,
        chunk_size=20,
        overlap=5,
    )

    results = service.search(
        "How is Python used?",
        top_k=1,
    )

    assert len(results) == 1
    assert "Python" in results[0].chunk.text