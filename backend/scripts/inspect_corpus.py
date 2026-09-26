from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document


ROOT_DIR = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT_DIR / "data" / "evaluation" / "corpus"

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


def main() -> None:
    all_chunk_count = 0

    for file_path in sorted(CORPUS_DIR.glob("*.md")):
        documents = load_document(file_path)

        chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        all_chunk_count += len(chunks)

        print("\n" + "=" * 80)
        print(file_path.name)
        print("=" * 80)

        for chunk in chunks:
            print(f"\n[{chunk.chunk_id}]")
            print(chunk.text)

    print("\n" + "=" * 80)
    print(f"TOTAL CHUNKS: {all_chunk_count}")
    print(
        f"CHUNK SIZE: {CHUNK_SIZE} words | "
        f"OVERLAP: {CHUNK_OVERLAP} words"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()