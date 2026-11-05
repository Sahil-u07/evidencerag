from pathlib import Path

import pytest

from app.ingestion.chunker import build_chunks, chunk_by_sentences, chunk_documents
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


def test_sentence_chunks_never_split_a_sentence():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    document = DocumentChunk(text=text, source="a.txt")

    chunks = chunk_by_sentences([document], max_words=6, overlap_sentences=0)

    assert [chunk.text for chunk in chunks] == [
        "Alpha beta gamma. Delta epsilon zeta.",
        "Eta theta iota.",
    ]


def test_sentence_chunks_overlap_by_whole_sentences():
    text = "One two three. Four five six. Seven eight nine."
    document = DocumentChunk(text=text, source="a.txt")

    chunks = chunk_by_sentences([document], max_words=6, overlap_sentences=1)

    assert chunks[1].text.startswith("Four five six.")


def test_paragraph_heading_stays_apart_from_following_sentence():
    text = "3.1. Heading\n\nBody sentence here."
    document = DocumentChunk(text=text, source="a.txt")

    chunks = chunk_by_sentences([document], max_words=2, overlap_sentences=0)

    assert chunks[0].text == "3.1. Heading"


def test_oversized_sentence_is_split_by_words():
    document = DocumentChunk(text=" ".join(["word"] * 25), source="a.txt")

    chunks = chunk_by_sentences([document], max_words=10, overlap_sentences=0)

    assert [len(chunk.text.split()) for chunk in chunks] == [10, 10, 5]


def test_sentence_chunk_ids_include_pages():
    document = DocumentChunk(text="Alpha beta.", source="a.pdf", page=3)

    chunks = chunk_by_sentences([document])

    assert chunks[0].chunk_id == "a.pdf:p3:1"
    assert chunks[0].page == 3


def test_overlap_at_or_above_max_cannot_loop_forever():
    text = "One two three four. Five six seven eight."
    document = DocumentChunk(text=text, source="a.txt")

    chunks = chunk_by_sentences([document], max_words=4, overlap_sentences=1)

    assert chunks[-1].text.endswith("Five six seven eight.")
    assert len(chunks) == 2


def test_build_chunks_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown chunking"):
        build_chunks([], "paragraphs", 100, 0)
