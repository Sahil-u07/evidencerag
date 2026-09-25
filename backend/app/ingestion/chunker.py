from app.ingestion.models import DocumentChunk


def chunk_documents(
    documents: list[DocumentChunk],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be greater than or equal to 0 "
            "and smaller than chunk_size"
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
                    chunk_id=f"{document.source}:{chunk_number}",
                    metadata=document.metadata.copy(),
                )
            )

            if end == len(words):
                break

            start = end - overlap

    return chunks