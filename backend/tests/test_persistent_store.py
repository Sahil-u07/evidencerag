import numpy as np
import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.persistent_store import PersistentIndexStore


def make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text="EvidenceRAG uses hybrid retrieval.",
            source="test.pdf",
            page=1,
            chunk_id="test.pdf:1",
            metadata={"file_type": "pdf", "page": "1"},
        ),
        DocumentChunk(
            text="Dense retrieval uses embeddings.",
            source="test.pdf",
            page=2,
            chunk_id="test.pdf:2",
            metadata={"file_type": "pdf", "page": "2"},
        ),
    ]


def test_store_does_not_exist_initially(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    assert store.exists() is False


def test_save_and_load_round_trip(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")
    chunks = make_chunks()
    embeddings = [
        [1.0, 0.0, 0.5],
        [0.2, 0.8, 0.1],
    ]

    store.save(chunks, embeddings)

    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == chunks
    assert loaded_embeddings.dtype == np.float32
    np.testing.assert_allclose(
        loaded_embeddings,
        np.asarray(embeddings, dtype=np.float32),
    )


def test_save_creates_expected_files(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(
        make_chunks(),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert (tmp_path / "index" / "index.json").exists()
    assert (tmp_path / "index" / "embeddings.npy").exists()


def test_save_rejects_mismatched_lengths(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="match"):
        store.save(
            make_chunks(),
            [[1.0, 0.0]],
        )


def test_save_rejects_non_2d_embeddings(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="2D"):
        store.save(
            make_chunks(),
            [1.0, 0.0],
        )


def test_load_missing_index_raises(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(FileNotFoundError):
        store.load()


def test_clear_removes_persisted_index(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(
        make_chunks(),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    store.clear()

    assert store.exists() is False

    with pytest.raises(FileNotFoundError):
        store.load()


def test_store_creates_storage_directory(tmp_path):
    storage_dir = tmp_path / "nested" / "index"
    store = PersistentIndexStore(storage_dir)

    store.save(
        make_chunks(),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert storage_dir.exists()
    assert store.exists() is True
def test_upsert_document_replaces_existing_document(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    old_chunks = [
        DocumentChunk(
            text="Old content",
            source="report.pdf",
            page=1,
            chunk_id="report.pdf:1",
        )
    ]

    store.save(
        old_chunks,
        [[1.0, 0.0]],
    )

    new_chunks = [
        DocumentChunk(
            text="New content page one",
            source="report.pdf",
            page=1,
            chunk_id="report.pdf:1",
        ),
        DocumentChunk(
            text="New content page two",
            source="report.pdf",
            page=2,
            chunk_id="report.pdf:2",
        ),
    ]

    store.upsert_document(
        "report.pdf",
        new_chunks,
        [[0.1, 0.9], [0.8, 0.2]],
    )

    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == new_chunks
    np.testing.assert_allclose(
        loaded_embeddings,
        np.asarray(
            [[0.1, 0.9], [0.8, 0.2]],
            dtype=np.float32,
        ),
    )


def test_upsert_document_preserves_other_documents(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    existing_chunks = [
        DocumentChunk(
            text="Keep this document",
            source="keep.pdf",
            page=1,
            chunk_id="keep.pdf:1",
        ),
        DocumentChunk(
            text="Replace this document",
            source="replace.pdf",
            page=1,
            chunk_id="replace.pdf:1",
        ),
    ]

    store.save(
        existing_chunks,
        [[1.0, 0.0], [0.0, 1.0]],
    )

    replacement = [
        DocumentChunk(
            text="Replacement content",
            source="replace.pdf",
            page=1,
            chunk_id="replace.pdf:1",
        )
    ]

    store.upsert_document(
        "replace.pdf",
        replacement,
        [[0.3, 0.7]],
    )

    loaded_chunks, loaded_embeddings = store.load()

    assert [chunk.source for chunk in loaded_chunks] == [
        "keep.pdf",
        "replace.pdf",
    ]

    assert loaded_chunks[0].text == "Keep this document"
    assert loaded_chunks[1].text == "Replacement content"

    np.testing.assert_allclose(
        loaded_embeddings,
        np.asarray(
            [[1.0, 0.0], [0.3, 0.7]],
            dtype=np.float32,
        ),
    )


def test_remove_document_removes_only_requested_document(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    chunks = [
        DocumentChunk(
            text="Keep",
            source="keep.pdf",
            page=1,
            chunk_id="keep.pdf:1",
        ),
        DocumentChunk(
            text="Remove",
            source="remove.pdf",
            page=1,
            chunk_id="remove.pdf:1",
        ),
    ]

    store.save(
        chunks,
        [[1.0, 0.0], [0.0, 1.0]],
    )

    removed = store.remove_document("remove.pdf")

    assert removed is True

    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == [chunks[0]]

    np.testing.assert_allclose(
        loaded_embeddings,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )


def test_remove_document_returns_false_when_missing(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(
        make_chunks(),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    removed = store.remove_document("missing.pdf")

    assert removed is False


def test_remove_last_document_clears_store(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(
        make_chunks(),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert store.remove_document("test.pdf") is True
    assert store.exists() is False


def test_upsert_rejects_chunks_from_different_source(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    chunks = [
        DocumentChunk(
            text="Wrong source",
            source="other.pdf",
            page=1,
            chunk_id="other.pdf:1",
        )
    ]

    with pytest.raises(ValueError, match="specified source"):
        store.upsert_document(
            "report.pdf",
            chunks,
            [[1.0, 0.0]],
        )
