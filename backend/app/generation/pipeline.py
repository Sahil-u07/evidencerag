from dataclasses import dataclass

from app.generation.generator import GeneratedAnswer, OpenAIGenerator
from app.generation.verifier import EvidenceVerifier, VerificationResult


@dataclass
class RAGResponse:
    query: str
    answer: GeneratedAnswer
    evidence: list
    verification: VerificationResult


class RAGPipeline:
    """
    End-to-end EvidenceRAG pipeline.

    Flow:
        Query
          ↓
        Retrieval
          ↓
        Reranking
          ↓
        Grounded generation
          ↓
        Evidence verification
    """

    def __init__(
        self,
        retriever,
        generator=None,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or OpenAIGenerator()
        self.verifier = verifier or EvidenceVerifier()

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        evidence = self.retriever.search(
            query,
            top_k=top_k,
        )

        answer = self.generator.generate(
            query=query,
            evidence=evidence,
        )

        verification = self.verifier.verify_citations(
            answer=answer.answer,
            evidence_count=len(evidence),
        )

        return RAGResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            verification=verification,
        )