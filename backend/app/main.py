from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.generation.ollama_generator import OllamaGenerator
from app.generation.pipeline import RAGPipeline
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever


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


class AskResponse(BaseModel):
    query: str
    answer: str
    evidence: list[EvidenceItem]
    verification: VerificationResponse


class ApplicationState:
    def __init__(self) -> None:
        self.pipeline: RAGPipeline | None = None
        self.retriever: RerankedHybridRetriever | None = None


state = ApplicationState()


def build_retriever() -> RerankedHybridRetriever:
    """
    Build the complete retrieval and reranking stack.

    Pipeline:

        Documents
            ↓
        Chunking
            ↓
        Embeddings
            ↓
        Dense Retrieval
            +
        BM25 Retrieval
            ↓
        Reciprocal Rank Fusion
            ↓
        Cross-Encoder Reranking
    """

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedder = TextEmbedder()

    documents = []

    source_directories = [
        EVALUATION_CORPUS,
        UPLOAD_DIR,
    ]

    for directory in source_directories:
        if not directory.exists():
            continue

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            documents.extend(
                load_document(path)
            )

    if not documents:
        raise RuntimeError(
            "No supported documents found in the configured "
            "document directories."
        )

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        overlap=10,
    )

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

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
                score=float(reranked_result.score),
                text=chunk.text,
            )
        )

    return items


@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Build the retrieval and generation pipeline when
    the API starts.
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
        cited_evidence=result.verification.cited_evidence,
        unsupported_claims=(
            result.verification.unsupported_claims
        ),
        reason=result.verification.reason,
    )

    return AskResponse(
        query=result.query,
        answer=result.answer.answer,
        evidence=evidence,
        verification=verification,
    )


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

    original_name = Path(file.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported types: PDF, TXT, MD."
            ),
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_name = (
        f"{uuid4().hex}_{original_name}"
    )

    destination = UPLOAD_DIR / unique_name

    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)

        new_retriever = build_retriever()

        new_pipeline = RAGPipeline(
            retriever=new_retriever,
            generator=OllamaGenerator(),
        )

        state.retriever = new_retriever
        state.pipeline = new_pipeline

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
        "message": "Document uploaded and indexed successfully.",
        "filename": original_name,
        "stored_as": unique_name,
    }