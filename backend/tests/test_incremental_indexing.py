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


class _FakeEmbedder:
    def __init__(self, model_name=None):
        pass

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, query):
        return [1.0, 0.0]


class _FakeReranker:
    def __init__(self, model_name=None):
        pass

    def rerank(self, query, results, top_k=5):
        return []


def _patch_main(monkeypatch, tmp_path):
    import app.main as main

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(main, "TextEmbedder", _FakeEmbedder)
    monkeypatch.setattr(main, "CrossEncoderReranker", _FakeReranker)
    monkeypatch.setattr(main, "get_index_storage_dir", lambda: tmp_path / "index")
    return main, uploads


def test_benchmark_corpus_is_not_indexed(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    upload = uploads / "mine.txt"
    upload.write_text("my document", encoding="utf-8")

    assert main.collect_source_paths() == [upload]


def test_app_starts_with_no_documents(monkeypatch, tmp_path):
    main, _ = _patch_main(monkeypatch, tmp_path)

    retriever = main.build_retriever()

    assert retriever.hybrid_retriever.dense_retriever.chunks == []
    assert retriever.hybrid_retriever.bm25_retriever.chunks == []


def test_deleting_last_document_empties_index(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    upload = uploads / "mine.txt"
    upload.write_text("my document text", encoding="utf-8")
    main.build_retriever()
    upload.unlink()

    retriever = main.build_retriever()

    assert retriever.hybrid_retriever.dense_retriever.chunks == []
    assert retriever.hybrid_retriever.bm25_retriever.chunks == []
    assert not (tmp_path / "index" / "index.json").exists()


def test_old_index_format_is_rebuilt(monkeypatch, tmp_path):
    import json

    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "mine.txt").write_text("my document text", encoding="utf-8")
    main.build_retriever()

    index_path = tmp_path / "index" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["version"] = 2
    payload["chunks"][0]["text"] = "stale sentinel text"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    retriever = main.build_retriever()

    rebuilt = json.loads(index_path.read_text(encoding="utf-8"))
    assert rebuilt["version"] == main.PersistentIndexStore.VERSION
    assert "stale sentinel text" not in index_path.read_text(encoding="utf-8")
    assert all(
        "stale sentinel text" not in chunk.text
        for chunk in retriever.hybrid_retriever.dense_retriever.chunks
    )


def test_production_uses_the_retrieval_defaults(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "mine.txt").write_text("one two three. four five six.", encoding="utf-8")
    calls = []
    real = main.build_chunks

    def spy(documents, method, size, overlap):
        calls.append((method, size, overlap))
        return real(documents, method, size, overlap)

    monkeypatch.setattr(main, "build_chunks", spy)
    main.build_retriever()

    defaults = main.retrieval_defaults
    assert calls == [(defaults.CHUNKING, defaults.CHUNK_SIZE, defaults.CHUNK_OVERLAP)]


def test_candidate_count_scales_with_top_k(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "mine.txt").write_text("one two three. four five six.", encoding="utf-8")
    retriever = main.build_retriever()
    requested = []
    real = retriever.hybrid_retriever.search

    def spy(query, top_k, candidate_k):
        requested.append(candidate_k)
        return real(query, top_k=top_k, candidate_k=candidate_k)

    monkeypatch.setattr(retriever.hybrid_retriever, "search", spy)
    retriever.search("one", top_k=5)
    retriever.search("one", top_k=20)

    assert requested == [main.retrieval_defaults.CANDIDATE_K, 80]


def test_app_starts_when_the_only_document_has_no_text(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "blank.txt").write_text("   \n  ", encoding="utf-8")

    retriever = main.build_retriever()

    assert retriever.hybrid_retriever.dense_retriever.chunks == []
    assert retriever.hybrid_retriever.bm25_retriever.chunks == []
    assert not (tmp_path / "index" / "index.json").exists()


def test_deleting_the_good_document_leaves_a_textless_one(monkeypatch, tmp_path):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    good = uploads / "good.txt"
    good.write_text("my document text", encoding="utf-8")
    (uploads / "blank.txt").write_text("   ", encoding="utf-8")
    assert main.build_retriever().hybrid_retriever.dense_retriever.chunks
    good.unlink()

    retriever = main.build_retriever()

    assert retriever.hybrid_retriever.dense_retriever.chunks == []
    assert not (tmp_path / "index" / "index.json").exists()


def test_a_corrupt_pdf_does_not_stop_a_full_build(monkeypatch, tmp_path, caplog):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "broken.pdf").write_bytes(b"this is not a pdf")
    (uploads / "good.txt").write_text("my document text", encoding="utf-8")

    retriever = main.build_retriever()

    assert [chunk.source for chunk in retriever.hybrid_retriever.dense_retriever.chunks] == ["good.txt"]
    assert "broken.pdf" in caplog.text


def test_a_corrupt_pdf_does_not_stop_an_incremental_build(monkeypatch, tmp_path, caplog):
    main, uploads = _patch_main(monkeypatch, tmp_path)
    (uploads / "good.txt").write_text("my document text", encoding="utf-8")
    main.build_retriever()
    (uploads / "broken.pdf").write_bytes(b"this is not a pdf")

    retriever = main.build_retriever()

    assert [chunk.source for chunk in retriever.hybrid_retriever.dense_retriever.chunks] == ["good.txt"]
    assert "broken.pdf" in caplog.text
