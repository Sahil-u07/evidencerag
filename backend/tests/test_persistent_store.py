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

    embeddings = [[1.0, 0.0, 0.5], [0.2, 0.8, 0.1]]

    store.save(chunks, embeddings)

    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == chunks
    assert loaded_embeddings.dtype == np.float32

    np.testing.assert_allclose(
        loaded_embeddings, np.asarray(embeddings, dtype=np.float32)
    )


def test_save_creates_expected_files(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    assert (tmp_path / "index" / "index.json").exists()

    assert (tmp_path / "index" / "embeddings.npy").exists()


def test_save_rejects_mismatched_lengths(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="match"):
        store.save(make_chunks(), [[1.0, 0.0]])


def test_save_rejects_non_2d_embeddings(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="2D"):
        store.save(make_chunks(), [1.0, 0.0])


def test_load_missing_index_raises(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    with pytest.raises(FileNotFoundError):
        store.load()


def test_clear_removes_persisted_index(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    store.save(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    store.clear()

    assert store.exists() is False

    with pytest.raises(FileNotFoundError):
        store.load()


def test_store_creates_storage_directory(tmp_path):
    storage_dir = tmp_path / "nested" / "index"

    store = PersistentIndexStore(storage_dir)

    store.save(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    assert storage_dir.exists()
    assert store.exists() is True


def test_manifest_round_trip(tmp_path):
    store = PersistentIndexStore(tmp_path / "index")

    manifest = {str((tmp_path / "report.pdf").resolve()): "abc123"}

    store.save(make_chunks(), [[1.0, 0.0], [0.0, 1.0]], manifest=manifest)

    assert store.load_manifest() == manifest

    assert store.matches_manifest(manifest) is True

    assert store.matches_manifest({}) is False
