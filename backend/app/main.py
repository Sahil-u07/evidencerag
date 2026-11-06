from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import indexing
from app.core.config import settings
from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.generation.streaming import describe_error, generator_available, stream_generate
from app.ingestion.loader import UnreadableDocumentError
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
    VerificationResponse,
    serialize_evidence,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("evidencerag.api")


class ApplicationState:
    def __init__(self) -> None:
        # The pipeline owns its retriever, so one assignment swaps both together.
        self.pipeline: RAGPipeline | None = None


state = ApplicationState()

# Rebuilds write the same index files, so they must not overlap.
rebuild_lock = threading.Lock()


def rebuild_pipeline() -> None:
    with rebuild_lock:
        state.pipeline = RAGPipeline(
            retriever=indexing.build_retriever(),
            generator=OllamaGenerator(),
        )


def index_upload(path: Path) -> None:
    try:
        has_text = indexing.has_indexable_text(path)
    except UnreadableDocumentError as exc:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read.",
        ) from exc

    # Check before indexing so a scanned PDF or blank file is refused, not stored.
    if not has_text:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this file.",
        )

    rebuild_pipeline()


def require_supported_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in indexing.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported types: PDF, TXT, MD.",
        )


def sse_event(
    event_type: str,
    payload: dict[str, Any],
) -> str:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


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
    pipeline = state.pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    try:
        result = pipeline.ask(
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
        documents=indexing.list_uploaded_documents(),
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

    require_supported_extension(safe_filename)

    destination = (
        indexing.UPLOAD_DIR / safe_filename
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

    require_supported_extension(original_name)

    max_upload_size = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    indexing.UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_name = (
        f"{uuid4().hex}_{original_name}"
    )

    destination = (
        indexing.UPLOAD_DIR / unique_name
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