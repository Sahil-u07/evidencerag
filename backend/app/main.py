from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core import retrieval_defaults
from app.core.config import settings
from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.generation.streaming import stream_generate
from app.ingestion.chunker import build_chunks
from app.ingestion.loader import UnreadableDocumentError, load_document
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

UPLOAD_DIR = PROJECT_ROOT / settings.upload_dir

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
        # The pipeline owns its retriever, so one assignment swaps both together.
        self.pipeline: RAGPipeline | None = None


state = ApplicationState()

# Rebuilds write the same index files, so they must not overlap.
rebuild_lock = threading.Lock()


def get_index_storage_dir() -> Path:
    return PROJECT_ROOT / settings.index_storage_dir


def collect_source_paths() -> list[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    return [
        path
        for path in sorted(UPLOAD_DIR.iterdir(), key=str)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def build_runtime_retriever(
    chunks,
    embeddings,
    embedder: TextEmbedder,
) -> RerankedHybridRetriever:
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
        documents.extend(
            load_readable_document(path)
        )

    chunks = build_chunks(
        documents,
        retrieval_defaults.CHUNKING,
        retrieval_defaults.CHUNK_SIZE,
        retrieval_defaults.CHUNK_OVERLAP,
    )

    if not chunks:
        return clear_textless_index(store)

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
        documents = load_readable_document(path)

        document_chunks = build_chunks(
            documents,
            retrieval_defaults.CHUNKING,
            retrieval_defaults.CHUNK_SIZE,
            retrieval_defaults.CHUNK_OVERLAP,
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
        return clear_textless_index(store)

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
    source_paths = collect_source_paths()
    index_storage_dir = get_index_storage_dir()
    store = PersistentIndexStore(index_storage_dir)
    embedder = TextEmbedder(retrieval_defaults.EMBEDDING_MODEL)

    if not source_paths:
        store.clear()
        return build_runtime_retriever([], [], embedder)

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
    with rebuild_lock:
        state.pipeline = RAGPipeline(
            retriever=build_retriever(),
            generator=OllamaGenerator(),
        )


def index_upload(path: Path) -> None:
    try:
        documents = load_document(path)
    except UnreadableDocumentError as exc:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read.",
        ) from exc

    # Check before indexing so a scanned PDF or blank file is refused, not stored.
    if not build_chunks(
        documents,
        retrieval_defaults.CHUNKING,
        retrieval_defaults.CHUNK_SIZE,
        retrieval_defaults.CHUNK_OVERLAP,
    ):
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file.",
        )

    rebuild_pipeline()


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


def sse_event(
    event_type: str,
    payload: dict[str, Any],
) -> str:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


def describe_error(exc: Exception) -> str:
    # ConnectTimeout is both of these, and it means Ollama could not be reached.
    if isinstance(exc, requests.ConnectionError):
        return (
            f"Could not reach Ollama at {settings.ollama_url.rstrip('/')}. "
            "Start Ollama and try again."
        )

    if isinstance(exc, requests.Timeout):
        return (
            "Ollama did not respond in time. "
            "It may still be loading the model; try again."
        )

    if (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code == 404
    ):
        return (
            f"The model {settings.ollama_model} is not available "
            f"in Ollama. Run: ollama pull {settings.ollama_model}"
        )

    return "Something went wrong while answering. Check the backend log."


def stream_query_events(
    query: str,
    top_k: int,
) -> Iterator[str]:
    # Read once, so a rebuild mid-request cannot mix two pipelines.
    pipeline = state.pipeline

    if pipeline is None:
        yield sse_event(
            "error",
            {
                "message": (
                    "Retrieval system is not ready."
                ),
            },
        )
        return

    started_at = time.perf_counter()

    try:
        # -----------------------------------------------------
        # Stage 1: Retrieval + reranking
        # -----------------------------------------------------
        retrieval_start = time.perf_counter()

        evidence = pipeline.retriever.search(
            query,
            top_k=top_k,
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        yield sse_event(
            "stage",
            {
                "i": 1,
                "ms": round(
                    retrieval_ms,
                    2,
                ),
            },
        )

        yield sse_event(
            "sources",
            {
                "items": [
                    item.model_dump()
                    for item in serialize_evidence(
                        evidence
                    )
                ],
            },
        )

        # -----------------------------------------------------
        # Stage 2: Generation
        # Tokens stream while this stage is active.
        # -----------------------------------------------------
        generation_start = time.perf_counter()

        answer_parts: list[str] = []
        truncated = False

        for token in stream_generate(
            query=query,
            evidence=evidence,
        ):
            truncated = truncated or token.truncated

            if not token.text:
                continue

            answer_parts.append(token.text)

            yield sse_event(
                "token",
                {
                    "t": token.text,
                },
            )

        generation_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        yield sse_event(
            "stage",
            {
                "i": 2,
                "ms": round(
                    generation_ms,
                    2,
                ),
            },
        )

        raw_answer = "".join(
            answer_parts
        ).strip()

        # -----------------------------------------------------
        # Stage 3: Citation alignment and grounding verification
        # -----------------------------------------------------
        verification_start = time.perf_counter()

        checked = pipeline.check_answer(
            raw_answer,
            evidence,
            truncated,
        )

        verification = checked.verification

        verification_ms = (
            time.perf_counter()
            - verification_start
        ) * 1000

        yield sse_event(
            "stage",
            {
                "i": 3,
                "ms": round(
                    verification_ms,
                    2,
                ),
            },
        )

        yield sse_event(
            "verification",
            {
                "supported": verification.supported,
                "cited_evidence": verification.cited_evidence,
                "unsupported_claims": verification.unsupported_claims,
                "reason": verification.reason,
            },
        )

        total_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        yield sse_event(
            "final",
            {
                "answer": checked.final_answer,
                "supported": verification.supported,
                "latency_ms": round(
                    total_ms,
                    2,
                ),
            },
        )

        logger.info(
            "Streaming RAG timing | total=%.0fms "
            "retrieval=%.0fms generation=%.0fms "
            "verification=%.0fms evidence=%d",
            total_ms,
            retrieval_ms,
            generation_ms,
            verification_ms,
            len(evidence),
        )

        yield sse_event(
            "done",
            {
                "latency_ms": round(
                    total_ms,
                    2,
                ),
                "supported": verification.supported,
            },
        )

    except Exception as exc:
        logger.exception(
            "Streaming RAG generation failed."
        )

        yield sse_event(
            "error",
            {
                "message": describe_error(exc),
            },
        )


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    rebuild_pipeline()

    yield

    state.pipeline = None


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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
            time.perf_counter()
            - start_time
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
        time.perf_counter()
        - start_time
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


def generator_available() -> bool:
    model = settings.ollama_model

    if ":" not in model:
        model += ":latest"

    try:
        response = requests.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags",
            timeout=2,
        )
        response.raise_for_status()

        return model in {
            item["name"] for item in response.json()["models"]
        }

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return False


@app.get("/health")
def health() -> dict[str, str | bool | int]:
    return {
        "status": "ok",
        "ready": state.pipeline is not None,
        "generator": "ollama",
        "model": settings.ollama_model,
        "generator_available": generator_available(),
        "max_upload_mb": settings.max_upload_size_mb,
    }


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
) -> SearchResponse:
    pipeline = state.pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval system is not ready.",
        )

    results = pipeline.retriever.search(
        request.query,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        results=serialize_evidence(
            results
        ),
    )


@app.post(
    "/query/stream",
)
def stream_query(
    request: AskRequest,
) -> StreamingResponse:
    return StreamingResponse(
        stream_query_events(
            request.query,
            request.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
        logger.exception("RAG generation failed.")

        raise HTTPException(
            status_code=503,
            detail=describe_error(exc),
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

    destination = (
        UPLOAD_DIR / safe_filename
    )

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
    except OSError as exc:
        logger.exception("Removing a document failed.")

        raise HTTPException(
            status_code=500,
            detail="The document could not be removed.",
        ) from exc

    try:
        rebuild_pipeline()

    except Exception as exc:
        logger.exception("Rebuilding the index after a delete failed.")

        # The old retriever still holds the deleted text, so stop serving it.
        state.pipeline = None

        raise HTTPException(
            status_code=500,
            detail=(
                "The document was removed, but the search index "
                "could not be rebuilt. Restart the backend to rebuild it."
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

        # Parsing and embedding take a while, and must not block other requests.
        await run_in_threadpool(index_upload, destination)

    except HTTPException:
        if destination.exists():
            destination.unlink()

        raise

    except Exception as exc:
        logger.exception("Indexing an upload failed.")

        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=500,
            detail="The document could not be indexed, so it was not saved.",
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