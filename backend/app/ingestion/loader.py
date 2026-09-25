from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import DocumentChunk


def load_text_file(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")

    return [
        DocumentChunk(
            text=text,
            source=file_path.name,
            metadata={"file_type": "text"},
        )
    ]


def load_pdf(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    reader = PdfReader(str(file_path))

    documents: list[DocumentChunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            DocumentChunk(
                text=text,
                source=file_path.name,
                page=page_number,
                metadata={
                    "file_type": "pdf",
                    "page": str(page_number),
                },
            )
        )

    return documents


def load_document(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix in {".txt", ".md"}:
        return load_text_file(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")