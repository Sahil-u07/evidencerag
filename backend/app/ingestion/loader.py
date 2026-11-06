from pathlib import Path

from pypdf import PdfReader

from app.ingestion.models import DocumentChunk


class UnreadableDocumentError(Exception):
    """The file exists but its contents cannot be parsed."""


def load_text_file(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)

    # Not utf-8-sig: rfc9110.txt starts with a BOM, and dropping it shifts every chunk.
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="cp1252", errors="replace")

    return [
        DocumentChunk(
            text=text,
            source=file_path.name,
            metadata={"file_type": "text"},
        )
    ]


def load_pdf(path: str | Path) -> list[DocumentChunk]:
    file_path = Path(path)

    # pypdf raises many error types on damaged files, not only PdfReadError.
    try:
        pages = [page.extract_text() or "" for page in PdfReader(str(file_path)).pages]
    except Exception as exc:
        raise UnreadableDocumentError(file_path.name) from exc

    documents: list[DocumentChunk] = []

    for page_number, text in enumerate(pages, start=1):
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