from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.ingestion.models import DocumentChunk


def test_chunk_documents_preserves_source():
    document = DocumentChunk(
        text="one two three four five six",
        source="example.txt",
    )

    chunks = chunk_documents(
        [document],
        chunk_size=3,
        overlap=1,
    )

    assert len(chunks) == 3
    assert all(chunk.source == "example.txt" for chunk in chunks)


def test_chunk_ids_are_unique():
    document = DocumentChunk(
        text="one two three four five six",
        source="example.txt",
    )

    chunks = chunk_documents(
        [document],
        chunk_size=3,
        overlap=1,
    )

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_load_text_document(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "EvidenceRAG processes documents for grounded question answering.",
        encoding="utf-8",
    )

    documents = load_document(file_path)

    assert len(documents) == 1
    assert documents[0].source == "sample.txt"
    assert documents[0].metadata["file_type"] == "text"
    assert "EvidenceRAG" in documents[0].text


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "sample.xyz"
    file_path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)