from typing import Any

from pydantic import BaseModel, Field


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


def serialize_evidence(evidence: list[Any]) -> list[EvidenceItem]:
    items = []

    for index, reranked_result in enumerate(evidence, start=1):
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
