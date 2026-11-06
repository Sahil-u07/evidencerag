import re

from app.core.text import split_sentences
from app.ingestion.models import DocumentChunk


def chunk_id(document: DocumentChunk, number: int) -> str:
    if document.page is None:
        return f"{document.source}:{number}"
    return f"{document.source}:p{document.page}:{number}"


def chunk_documents(
    documents: list[DocumentChunk], chunk_size: int = 800, overlap: int = 120
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to 0 and smaller than chunk_size"
        )

    chunks: list[DocumentChunk] = []

    for document in documents:
        words = document.text.split()
        start = 0
        chunk_number = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]

            chunk_number += 1

            chunks.append(
                DocumentChunk(
                    text=" ".join(chunk_words),
                    source=document.source,
                    page=document.page,
                    chunk_id=chunk_id(document, chunk_number),
                    metadata=document.metadata.copy(),
                )
            )

            if end == len(words):
                break

            start = end - overlap

    return chunks


def _sentences_of(text: str, max_words: int) -> list[str]:
    sentences: list[str] = []

    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in split_sentences(" ".join(paragraph.split())):
            words = sentence.split()
            sentences.extend(
                " ".join(words[start : start + max_words])
                for start in range(0, len(words), max_words)
            )

    return sentences


def chunk_by_sentences(
    documents: list[DocumentChunk], max_words: int = 200, overlap_sentences: int = 1
) -> list[DocumentChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be 0 or greater")

    chunks: list[DocumentChunk] = []

    for document in documents:
        current: list[str] = []
        word_count = 0
        number = 0

        def flush() -> None:
            nonlocal number
            number += 1
            chunks.append(
                DocumentChunk(
                    text=" ".join(current),
                    source=document.source,
                    page=document.page,
                    chunk_id=chunk_id(document, number),
                    metadata=document.metadata.copy(),
                )
            )

        for sentence in _sentences_of(document.text, max_words):
            size = len(sentence.split())

            if current and word_count + size > max_words:
                flush()
                kept = current[-overlap_sentences:] if overlap_sentences else []
                kept_size = sum(len(item.split()) for item in kept)
                # Keeping the overlap would leave no room, so drop it to make progress.
                current = kept if kept_size + size <= max_words else []
                word_count = kept_size if current else 0

            current.append(sentence)
            word_count += size

        if current:
            flush()

    return chunks


def build_chunks(
    documents: list[DocumentChunk], method: str, size: int, overlap: int
) -> list[DocumentChunk]:
    if method == "words":
        return chunk_documents(documents, chunk_size=size, overlap=overlap)
    if method == "sentences":
        return chunk_by_sentences(documents, max_words=size, overlap_sentences=overlap)
    raise ValueError(f"Unknown chunking method: {method}")
