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


def test_chunk_ids_are_unique_across_pdf_pages():
    pages = [
        DocumentChunk(text="alpha beta gamma", source="a.pdf", page=1),
        DocumentChunk(text="delta epsilon zeta", source="a.pdf", page=2),
    ]

    chunks = chunk_documents(pages, chunk_size=3, overlap=0)

    assert [chunk.chunk_id for chunk in chunks] == ["a.pdf:p1:1", "a.pdf:p2:1"]


def test_unpaged_chunk_ids_stay_stable_for_v1_labels():
    document = DocumentChunk(text="alpha beta gamma", source="a.md")

    chunks = chunk_documents([document], chunk_size=3, overlap=0)

    assert chunks[0].chunk_id == "a.md:1"
