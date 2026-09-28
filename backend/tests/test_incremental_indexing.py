from pathlib import Path

import numpy as np

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.main import update_persistent_index_incrementally
from app.retrieval.persistent_store import PersistentIndexStore


class FakeEmbedder:
    def encode(self, texts):
        return np.ones(
            (len(texts), 3),
            dtype=np.float32,
        )


def build_chunks(path: Path):
    documents = load_document(path)

    return chunk_documents(
        documents,
        chunk_size=50,
        overlap=10,
    )


def test_incremental_index_adds_new_document(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    original = corpus_dir / "original.md"
    original.write_text(
        "# Original\n\nOriginal document content.",
        encoding="utf-8",
    )

    new_document = corpus_dir / "new_document.md"
    new_document.write_text(
        "# New Document\n\nNew incremental content.",
        encoding="utf-8",
    )

    store = PersistentIndexStore(tmp_path / "index")

    original_chunks = build_chunks(original)
    original_embeddings = np.ones(
        (len(original_chunks), 3),
        dtype=np.float32,
    )

    initial_paths = [original]
    initial_manifest = store.build_manifest(initial_paths)

    store.save(
        original_chunks,
        original_embeddings,
        manifest=initial_manifest,
    )

    existing_chunks, existing_embeddings = store.load()

    updated_paths = [
        original,
        new_document,
    ]

    updated_manifest = store.build_manifest(
        updated_paths
    )

    update_persistent_index_incrementally(
        store=store,
        source_paths=updated_paths,
        current_manifest=updated_manifest,
        previous_manifest=initial_manifest,
        existing_chunks=existing_chunks,
        existing_embeddings=existing_embeddings,
        embedder=FakeEmbedder(),
    )

    chunks, embeddings = store.load()

    sources = {
        chunk.source
        for chunk in chunks
    }

    assert original.name in sources
    assert new_document.name in sources
    assert embeddings.shape[0] == len(chunks)
    assert store.matches_manifest(
        updated_manifest
    )


def test_incremental_index_removes_deleted_document(
    tmp_path,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    first = corpus_dir / "first.md"
    second = corpus_dir / "second.md"

    first.write_text(
        "# First\n\nFirst document content.",
        encoding="utf-8",
    )

    second.write_text(
        "# Second\n\nSecond document content.",
        encoding="utf-8",
    )

    store = PersistentIndexStore(tmp_path / "index")

    initial_paths = [
        first,
        second,
    ]

    initial_chunks = (
        build_chunks(first)
        + build_chunks(second)
    )

    initial_embeddings = np.ones(
        (len(initial_chunks), 3),
        dtype=np.float32,
    )

    initial_manifest = store.build_manifest(
        initial_paths
    )

    store.save(
        initial_chunks,
        initial_embeddings,
        manifest=initial_manifest,
    )

    first.unlink()

    remaining_paths = [second]

    current_manifest = store.build_manifest(
        remaining_paths
    )

    existing_chunks, existing_embeddings = (
        store.load()
    )

    update_persistent_index_incrementally(
        store=store,
        source_paths=remaining_paths,
        current_manifest=current_manifest,
        previous_manifest=initial_manifest,
        existing_chunks=existing_chunks,
        existing_embeddings=existing_embeddings,
        embedder=FakeEmbedder(),
    )

    chunks, embeddings = store.load()

    sources = {
        chunk.source
        for chunk in chunks
    }

    assert first.name not in sources
    assert second.name in sources
    assert embeddings.shape[0] == len(chunks)
    assert store.matches_manifest(
        current_manifest
    )


def test_incremental_index_preserves_existing_embeddings(
    tmp_path,
):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    original = corpus_dir / "original.md"
    original.write_text(
        "# Original\n\nOriginal document content.",
        encoding="utf-8",
    )

    new_document = corpus_dir / "new.md"
    new_document.write_text(
        "# New\n\nNew document content.",
        encoding="utf-8",
    )

    store = PersistentIndexStore(tmp_path / "index")

    original_chunks = build_chunks(original)

    original_embeddings = np.full(
        (len(original_chunks), 3),
        7.0,
        dtype=np.float32,
    )

    initial_manifest = store.build_manifest(
        [original]
    )

    store.save(
        original_chunks,
        original_embeddings,
        manifest=initial_manifest,
    )

    existing_chunks, existing_embeddings = (
        store.load()
    )

    updated_paths = [
        original,
        new_document,
    ]

    updated_manifest = store.build_manifest(
        updated_paths
    )

    update_persistent_index_incrementally(
        store=store,
        source_paths=updated_paths,
        current_manifest=updated_manifest,
        previous_manifest=initial_manifest,
        existing_chunks=existing_chunks,
        existing_embeddings=existing_embeddings,
        embedder=FakeEmbedder(),
    )

    chunks, embeddings = store.load()

    original_indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.source == original.name
    ]

    new_indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.source == new_document.name
    ]

    assert original_indices
    assert new_indices

    assert np.all(
        embeddings[original_indices] == 7.0
    )

    assert np.all(
        embeddings[new_indices] == 1.0
    )