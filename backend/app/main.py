from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.persistent_store import PersistentIndexStore
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("evidencerag.api")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "corpus"
)

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceItem(BaseModel):
    evidence_id: int
    source: str
    chunk_id: str
    page: int | None
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[EvidenceItem]


class VerificationResponse(BaseModel):
    supported: bool
    cited_evidence: list[int]
    unsupported_claims: list[str]
    reason: str


class MetricsResponse(BaseModel):
    query: str
    top_k: int
    retrieved_evidence_count: int
    cited_evidence_count: int
    verification_supported: bool
    latency_ms: float


class AskResponse(BaseModel):
    query: str
    answer: str
    evidence: list[EvidenceItem]
    verification: VerificationResponse
    metrics: MetricsResponse


class DocumentItem(BaseModel):
    filename: str
    extension: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class ApplicationState:
    def __init__(self) -> None:
        self.pipeline: RAGPipeline | None = None
        self.retriever: RerankedHybridRetriever | None = None


state = ApplicationState()


def get_index_storage_dir() -> Path:
    """Resolve the configured persistent index directory."""
    index_storage_dir = Path(
        settings.index_storage_dir
    )

    if not index_storage_dir.is_absolute():
        index_storage_dir = (
            PROJECT_ROOT / index_storage_dir
        )

    return index_storage_dir


def collect_source_paths() -> list[Path]:
    """Collect all supported source documents."""
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_paths: list[Path] = []

    source_directories = [
        EVALUATION_CORPUS,
        UPLOAD_DIR,
    ]

    for directory in source_directories:
        if not directory.exists():
            continue

        for path in sorted(
            directory.iterdir(),
            key=lambda item: str(item),
        ):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            source_paths.append(path)

    return source_paths


def build_runtime_retriever(
    chunks,
    embeddings,
    embedder: TextEmbedder,
) -> RerankedHybridRetriever:
    """Build the in-memory retrieval and reranking stack."""
    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    hybrid_retriever.add(
        chunks,
        embeddings,
    )

    reranker = CrossEncoderReranker()

    return RerankedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )


def build_full_persistent_index(
    store: PersistentIndexStore,
    source_paths: list[Path],
    manifest: dict[str, str],
    embedder: TextEmbedder,
):
    """Parse, chunk, embed, and persist the complete corpus."""
    documents = []

    for path in source_paths:
        documents.extend(
            load_document(path)
        )

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        overlap=10,
    )

    if not chunks:
        raise RuntimeError(
            "Documents produced no indexable chunks."
        )

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

    store.save(
        chunks,
        embeddings,
        manifest=manifest,
    )

    logger.info(
        "Built and persisted index "
        "(chunks=%d, files=%d)",
        len(chunks),
        len(manifest),
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
    """
    Incrementally update the persistent index.

    Only added or modified source files are parsed,
    chunked, and embedded. Deleted files are removed
    from the persisted index.
    """
    changed_paths = [
        path
        for path in source_paths
        if previous_manifest.get(
            str(path.resolve())
        )
        != current_manifest.get(
            str(path.resolve())
        )
    ]

    deleted_paths = [
        Path(path)
        for path in previous_manifest
        if path not in current_manifest
    ]

    sources_to_replace = {
        path.name
        for path in changed_paths
    }

    sources_to_replace.update(
        path.name
        for path in deleted_paths
    )

    kept_indices = [
        index
        for index, chunk in enumerate(
            existing_chunks
        )
        if chunk.source not in sources_to_replace
    ]

    kept_chunks = [
        existing_chunks[index]
        for index in kept_indices
    ]

    if kept_indices:
        kept_embeddings = existing_embeddings[
            kept_indices
        ]
    else:
        kept_embeddings = np.empty(
            (
                0,
                existing_embeddings.shape[1],
            ),
            dtype=np.float32,
        )

    new_chunks = []
    new_embedding_blocks = []

    for path in changed_paths:
        documents = load_document(path)

        document_chunks = chunk_documents(
            documents,
            chunk_size=50,
            overlap=10,
        )

        if not document_chunks:
            logger.warning(
                "Document produced no indexable chunks: %s",
                path,
            )
            continue

        document_embeddings = embedder.encode(
            [
                chunk.text
                for chunk in document_chunks
            ]
        )

        document_embeddings = np.asarray(
            document_embeddings,
            dtype=np.float32,
        )

        if document_embeddings.ndim != 2:
            raise ValueError(
                "Document embeddings must be a 2D matrix"
            )

        if (
            len(document_embeddings)
            != len(document_chunks)
        ):
            raise ValueError(
                "Document chunks and embeddings "
                "are out of sync"
            )

        if (
            len(kept_embeddings) > 0
            and (
                kept_embeddings.shape[1]
                != document_embeddings.shape[1]
            )
        ):
            raise ValueError(
                "Embedding dimensions do not match"
            )

        new_chunks.extend(
            document_chunks
        )

        new_embedding_blocks.append(
            document_embeddings
        )

    combined_chunks = (
        kept_chunks + new_chunks
    )

    embedding_blocks = []

    if len(kept_embeddings) > 0:
        embedding_blocks.append(
            kept_embeddings
        )

    embedding_blocks.extend(
        new_embedding_blocks
    )

    if embedding_blocks:
        combined_embeddings = np.vstack(
            embedding_blocks
        )
    else:
        combined_embeddings = np.empty(
            (
                0,
                existing_embeddings.shape[1],
            ),
            dtype=np.float32,
        )

    if not combined_chunks:
        raise RuntimeError(
            "Documents produced no indexable chunks."
        )

    store.save(
        combined_chunks,
        combined_embeddings,
        manifest=current_manifest,
    )

    logger.info(
        "Incrementally updated persistent index "
        "(added_or_modified=%d, deleted=%d, "
        "chunks=%d, files=%d)",
        len(changed_paths),
        len(deleted_paths),
        len(combined_chunks),
        len(current_manifest),
    )

    return (
        combined_chunks,
        combined_embeddings,
    )


def build_retriever() -> RerankedHybridRetriever:
    """
    Build the complete retrieval and reranking stack.

    Uses a persistent local index when possible.

    Startup behavior:

        Matching manifest
            ↓
        Load persisted index

        Changed manifest
            ↓
        Incremental update
            ↓
        Persist updated index

        Missing/corrupt index
            ↓
        Full index rebuild

    Pipeline:

        Documents
            ↓
        Persistent Index
            ↓
        Dense Retrieval
            +
        BM25 Retrieval
            ↓
        Reciprocal Rank Fusion
            ↓
        Cross-Encoder Reranking
    """
    source_paths = collect_source_paths()

    if not source_paths:
        raise RuntimeError(
            "No supported documents found in the configured "
            "document directories."
        )

    index_storage_dir = (
        get_index_storage_dir()
    )

    store = PersistentIndexStore(
        index_storage_dir
    )

    embedder = TextEmbedder()

    current_manifest = store.build_manifest(
        source_paths
    )

    if store.matches_manifest(
        current_manifest
    ):
        try:
            chunks, embeddings = store.load()

            logger.info(
                "Loaded persistent index from %s "
                "(chunks=%d, files=%d)",
                index_storage_dir,
                len(chunks),
                len(current_manifest),
            )

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            logger.warning(
                "Persistent index could not be loaded. "
                "Rebuilding index."
            )

            chunks, embeddings = (
                build_full_persistent_index(
                    store,
                    source_paths,
                    current_manifest,
                    embedder,
                )
            )

    elif store.exists():
        try:
            (
                existing_chunks,
                existing_embeddings,
            ) = store.load()

            previous_manifest = (
                store.load_manifest()
            )

            chunks, embeddings = (
                update_persistent_index_incrementally(
                    store=store,
                    source_paths=source_paths,
                    current_manifest=current_manifest,
                    previous_manifest=previous_manifest,
                    existing_chunks=existing_chunks,
                    existing_embeddings=existing_embeddings,
                    embedder=embedder,
                )
            )

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            KeyError,
        ):
            logger.warning(
                "Persistent index could not be incrementally "
                "updated. Rebuilding full index."
            )

            chunks, embeddings = (
                build_full_persistent_index(
                    store,
                    source_paths,
                    current_manifest,
                    embedder,
                )
            )

    else:
        chunks, embeddings = (
            build_full_persistent_index(
                store,
                source_paths,
                current_manifest,
                embedder,
            )
        )

    return build_runtime_retriever(
        chunks,
        embeddings,
        embedder,
    )


def rebuild_pipeline() -> None:
    """Refresh the retrieval and RAG pipeline."""
    new_retriever = build_retriever()

    new_pipeline = RAGPipeline(
        retriever=new_retriever,
        generator=OllamaGenerator(),
    )

    state.retriever = new_retriever
    state.pipeline = new_pipeline


def serialize_evidence(
    evidence: list[Any],
) -> list[EvidenceItem]:
    items = []

    for index, reranked_result in enumerate(
        evidence,
        start=1,
    ):
        chunk = reranked_result.result.chunk

        items.append(
            EvidenceItem(
                evidence_id=index,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                score=float(
                    reranked_result.score
                ),
                text=chunk.text,
            )
        )

    return items


def list_uploaded_documents() -> list[DocumentItem]:
    """Return metadata for supported files in the upload directory."""
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = []

    for path in sorted(
        UPLOAD_DIR.iterdir()
    ):
        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        documents.append(
            DocumentItem(
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
            )
        )

    return documents


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Build the retrieval and generation pipeline
    when the API starts.
    """
    state.retriever = build_retriever()

    state.pipeline = RAGPipeline(
        retriever=state.retriever,
        generator=OllamaGenerator(),
    )

    yield

    state.pipeline = None
    state.retriever = None


app = FastAPI(
    title="EvidenceRAG API",
    description=(
        "Production-oriented document intelligence API "
        "with hybrid retrieval, reranking, grounded generation, "
        "and evidence verification."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "request_id=%s method=%s path=%s "
            "status=500 duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        raise

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = (
        request_id
    )

    logger.info(
        "request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "EvidenceRAG",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ready": (
            state.pipeline is not None
            and state.retriever is not None
        ),
        "generator": "ollama",
        "model": "llama3.2:3b",
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
) -> SearchResponse:
    if state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval system is not ready.",
        )

    results = state.retriever.search(
        request.query,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        results=serialize_evidence(results),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    try:
        result = state.pipeline.ask(
            request.query,
            top_k=request.top_k,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG generation failed. "
                "Make sure Ollama is running and "
                "llama3.2:3b is available."
            ),
        ) from exc

    evidence = serialize_evidence(
        result.evidence
    )

    verification = VerificationResponse(
        supported=result.verification.supported,
        cited_evidence=(
            result.verification.cited_evidence
        ),
        unsupported_claims=(
            result.verification.unsupported_claims
        ),
        reason=result.verification.reason,
    )

    metrics = MetricsResponse(
        query=result.metrics.query,
        top_k=result.metrics.top_k,
        retrieved_evidence_count=(
            result.metrics.retrieved_evidence_count
        ),
        cited_evidence_count=(
            result.metrics.cited_evidence_count
        ),
        verification_supported=(
            result.metrics.verification_supported
        ),
        latency_ms=result.metrics.latency_ms,
    )

    return AskResponse(
        query=result.query,
        answer=result.answer.answer,
        evidence=evidence,
        verification=verification,
        metrics=metrics,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=list_uploaded_documents(),
    )


@app.delete(
    "/documents/{filename}",
)
def delete_document(
    filename: str,
) -> dict[str, str]:
    safe_filename = Path(filename).name

    if safe_filename != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = (
        Path(safe_filename)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported types: PDF, TXT, MD."
            ),
        )

    destination = UPLOAD_DIR / safe_filename

    if (
        not destination.exists()
        or not destination.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:
        destination.unlink()

        rebuild_pipeline()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document was deleted, but rebuilding "
                "the retrieval index failed."
            ),
        ) from exc

    return {
        "message": (
            "Document deleted and index rebuilt successfully."
        ),
        "filename": safe_filename,
    }


@app.post(
    "/documents/upload",
)
async def upload_document(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    original_name = Path(
        file.filename
    ).name

    extension = (
        Path(original_name)
        .suffix
        .lower()
    )

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported types: PDF, TXT, MD."
            ),
        )

    max_upload_size = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_name = (
        f"{uuid4().hex}_{original_name}"
    )

    destination = (
        UPLOAD_DIR / unique_name
    )

    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(
                1024 * 1024
            ):
                total_bytes += len(chunk)

                if (
                    total_bytes
                    > max_upload_size
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File is too large. "
                            f"Maximum upload size is "
                            f"{settings.max_upload_size_mb} MB."
                        ),
                    )

                output.write(chunk)

        rebuild_pipeline()

    except HTTPException:
        if destination.exists():
            destination.unlink()

        raise

    except Exception as exc:
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "Document upload succeeded, but rebuilding "
                "the retrieval index failed."
            ),
        ) from exc

    finally:
        await file.close()

    return {
        "message": (
            "Document uploaded and indexed successfully."
        ),
        "filename": original_name,
        "stored_as": unique_name,
    }