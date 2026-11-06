import json
import logging
from pathlib import Path

import numpy as np

from app.core import retrieval_defaults
from app.core.config import settings
from app.ingestion.chunker import build_chunks
from app.ingestion.loader import UnreadableDocumentError, load_document
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.persistent_store import PersistentIndexStore
from app.schemas import DocumentItem


logger = logging.getLogger("evidencerag.indexing")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = PROJECT_ROOT / settings.upload_dir

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_index_storage_dir() -> Path:
    return PROJECT_ROOT / settings.index_storage_dir


def collect_source_paths() -> list[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    return [
        path
        for path in sorted(UPLOAD_DIR.iterdir(), key=str)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def list_uploaded_documents() -> list[DocumentItem]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(
            DocumentItem(
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
            )
        )

    return documents


def has_indexable_text(path: Path) -> bool:
    return bool(
        build_chunks(
            load_document(path),
            retrieval_defaults.CHUNKING,
            retrieval_defaults.CHUNK_SIZE,
            retrieval_defaults.CHUNK_OVERLAP,
        )
    )


def build_runtime_retriever(
    chunks, embeddings, embedder: TextEmbedder
) -> RerankedHybridRetriever:
    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    hybrid_retriever.add(chunks, embeddings)

    reranker = CrossEncoderReranker(retrieval_defaults.RERANKER_MODEL)

    return RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        minimum_candidates=retrieval_defaults.CANDIDATE_K,
    )


def load_readable_document(path: Path) -> list:
    try:
        return load_document(path)
    except UnreadableDocumentError:
        logger.warning("Skipping a file that could not be read: %s", path)
        return []


def clear_textless_index(store: PersistentIndexStore):
    logger.warning(
        "The uploaded documents contain no extractable text. "
        "Starting with an empty index."
    )
    store.clear()

    return [], np.empty((0, 0), dtype=np.float32)


def build_full_persistent_index(
    store: PersistentIndexStore,
    source_paths: list[Path],
    manifest: dict[str, str],
    embedder: TextEmbedder,
):
    documents = []

    for path in source_paths:
        documents.extend(load_readable_document(path))

    chunks = build_chunks(
        documents,
        retrieval_defaults.CHUNKING,
        retrieval_defaults.CHUNK_SIZE,
        retrieval_defaults.CHUNK_OVERLAP,
    )

    if not chunks:
        return clear_textless_index(store)

    embeddings = embedder.encode([chunk.text for chunk in chunks])

    store.save(chunks, embeddings, manifest=manifest)

    logger.info(
        "Built and persisted index (chunks=%d, files=%d)", len(chunks), len(manifest)
    )

    return chunks, embeddings


def update_persistent_index_incrementally(
    store: PersistentIndexStore,
    source_paths: list[Path],
    current_manifest: dict[str, str],
    previous_manifest: dict[str, str],
    existing_chunks,
    existing_embeddings,
    embedder: TextEmbedder,
):
    changed_paths = [
        path
        for path in source_paths
        if previous_manifest.get(str(path.resolve()))
        != current_manifest.get(str(path.resolve()))
    ]

    deleted_paths = [
        Path(path) for path in previous_manifest if path not in current_manifest
    ]

    sources_to_replace = {path.name for path in changed_paths}

    sources_to_replace.update(path.name for path in deleted_paths)

    kept_indices = [
        index
        for index, chunk in enumerate(existing_chunks)
        if chunk.source not in sources_to_replace
    ]

    kept_chunks = [existing_chunks[index] for index in kept_indices]

    if kept_indices:
        kept_embeddings = existing_embeddings[kept_indices]
    else:
        kept_embeddings = np.empty((0, existing_embeddings.shape[1]), dtype=np.float32)

    new_chunks = []
    new_embedding_blocks = []

    for path in changed_paths:
        documents = load_readable_document(path)

        document_chunks = build_chunks(
            documents,
            retrieval_defaults.CHUNKING,
            retrieval_defaults.CHUNK_SIZE,
            retrieval_defaults.CHUNK_OVERLAP,
        )

        if not document_chunks:
            logger.warning("Document produced no indexable chunks: %s", path)
            continue

        document_embeddings = embedder.encode([chunk.text for chunk in document_chunks])

        document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

        if document_embeddings.ndim != 2:
            raise ValueError("Document embeddings must be a 2D matrix")

        if len(document_embeddings) != len(document_chunks):
            raise ValueError("Document chunks and embeddings are out of sync")

        if len(kept_embeddings) > 0 and (
            kept_embeddings.shape[1] != document_embeddings.shape[1]
        ):
            raise ValueError("Embedding dimensions do not match")

        new_chunks.extend(document_chunks)

        new_embedding_blocks.append(document_embeddings)

    combined_chunks = kept_chunks + new_chunks

    embedding_blocks = []

    if len(kept_embeddings) > 0:
        embedding_blocks.append(kept_embeddings)

    embedding_blocks.extend(new_embedding_blocks)

    if embedding_blocks:
        combined_embeddings = np.vstack(embedding_blocks)
    else:
        combined_embeddings = np.empty(
            (0, existing_embeddings.shape[1]), dtype=np.float32
        )

    if not combined_chunks:
        return clear_textless_index(store)

    store.save(combined_chunks, combined_embeddings, manifest=current_manifest)

    logger.info(
        "Incrementally updated persistent index "
        "(added_or_modified=%d, deleted=%d, "
        "chunks=%d, files=%d)",
        len(changed_paths),
        len(deleted_paths),
        len(combined_chunks),
        len(current_manifest),
    )

    return (combined_chunks, combined_embeddings)


def build_retriever() -> RerankedHybridRetriever:
    source_paths = collect_source_paths()
    index_storage_dir = get_index_storage_dir()
    store = PersistentIndexStore(index_storage_dir)
    embedder = TextEmbedder(retrieval_defaults.EMBEDDING_MODEL)

    if not source_paths:
        store.clear()
        return build_runtime_retriever([], [], embedder)

    current_manifest = store.build_manifest(source_paths)

    if store.matches_manifest(current_manifest):
        try:
            chunks, embeddings = store.load()

            logger.info(
                "Loaded persistent index from %s (chunks=%d, files=%d)",
                index_storage_dir,
                len(chunks),
                len(current_manifest),
            )

        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Persistent index could not be loaded. Rebuilding index.")

            chunks, embeddings = build_full_persistent_index(
                store, source_paths, current_manifest, embedder
            )

    elif store.exists():
        try:
            (existing_chunks, existing_embeddings) = store.load()

            previous_manifest = store.load_manifest()

            chunks, embeddings = update_persistent_index_incrementally(
                store=store,
                source_paths=source_paths,
                current_manifest=current_manifest,
                previous_manifest=previous_manifest,
                existing_chunks=existing_chunks,
                existing_embeddings=existing_embeddings,
                embedder=embedder,
            )

        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            logger.warning(
                "Persistent index could not be incrementally "
                "updated. Rebuilding full index."
            )

            chunks, embeddings = build_full_persistent_index(
                store, source_paths, current_manifest, embedder
            )

    else:
        chunks, embeddings = build_full_persistent_index(
            store, source_paths, current_manifest, embedder
        )

    return build_runtime_retriever(chunks, embeddings, embedder)
