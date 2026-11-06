from dataclasses import dataclass
import logging
import re
import time

from app.core.text import split_sentences
from app.generation.ollama_generator import GeneratedAnswer
from app.generation.streaming import ABSTENTION_MESSAGE
from app.generation.verifier import EvidenceVerifier, VerificationResult


logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    query: str
    top_k: int
    retrieved_evidence_count: int
    cited_evidence_count: int
    verification_supported: bool
    latency_ms: float


@dataclass
class RAGResponse:
    query: str
    answer: GeneratedAnswer
    evidence: list
    verification: VerificationResult
    metrics: PipelineMetrics


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
        Citation normalization
          ↓
        Deterministic citation alignment
          ↓
        Grounding verification
          ↓
        Metrics collection
          ↓
        Verified answer or abstention
    """

    CITATION_PATTERN = re.compile(
        r"\[\s*Evidence\s*(\d+)\s*\]",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever,
        generator,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier or EvidenceVerifier()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[a-zA-Z0-9]+",
                text.lower(),
            )
            if len(token) > 2
        }

    @classmethod
    def _normalize_citations(cls, answer: str) -> str:
        def replace(match: re.Match) -> str:
            evidence_id = match.group(1)
            return f"[Evidence {evidence_id}]"

        return cls.CITATION_PATTERN.sub(
            replace,
            answer,
        )

    def _attach_citation(
        self,
        answer: str,
        evidence: list,
    ) -> str:
        if not answer.strip() or not evidence:
            return answer

        answer = self._normalize_citations(answer)

        cited_lines = []

        for line in answer.splitlines():
            sentences = [
                self._cite_sentence(sentence, evidence)
                for sentence in split_sentences(line)
            ]

            if sentences:
                cited_lines.append(" ".join(sentences))

        return "\n".join(cited_lines)

    def _cite_sentence(self, sentence: str, evidence: list) -> str:
        if self.CITATION_PATTERN.search(sentence):
            return self._normalize_citations(sentence)

        claim_tokens = self._tokens(sentence)

        if not claim_tokens:
            return sentence

        best_index = None
        best_score = 0.0

        for index, reranked_result in enumerate(evidence, start=1):
            evidence_tokens = self._tokens(reranked_result.result.chunk.text)

            if not evidence_tokens:
                continue

            score = len(claim_tokens & evidence_tokens) / len(claim_tokens)

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= 0.50:
            return f"{sentence} [Evidence {best_index}]"

        return sentence

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.perf_counter()

        # ---------------------------------------------------------
        # Stage 1: Retrieval + reranking
        # ---------------------------------------------------------
        retrieval_start = time.perf_counter()

        evidence = self.retriever.search(
            query,
            top_k=top_k,
        )

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        # ---------------------------------------------------------
        # Stage 2: Generation
        # ---------------------------------------------------------
        generation_start = time.perf_counter()

        answer = self.generator.generate(
            query=query,
            evidence=evidence,
        )

        generation_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        # ---------------------------------------------------------
        # Stage 3: Citation alignment
        # ---------------------------------------------------------
        citation_start = time.perf_counter()

        cited_answer = self._attach_citation(
            answer.answer,
            evidence,
        )

        answer = GeneratedAnswer(
            answer=cited_answer,
        )

        citation_ms = (
            time.perf_counter() - citation_start
        ) * 1000

        # ---------------------------------------------------------
        # Stage 4: Grounding verification
        # ---------------------------------------------------------
        verification_start = time.perf_counter()

        verification = self.verifier.verify_grounding(
            answer=answer.answer,
            evidence=evidence,
        )

        verification_ms = (
            time.perf_counter() - verification_start
        ) * 1000

        cited_evidence_count = len(
            verification.cited_evidence
        )

        total_ms = (
            time.perf_counter() - start_time
        ) * 1000

        metrics = PipelineMetrics(
            query=query,
            top_k=top_k,
            retrieved_evidence_count=len(evidence),
            cited_evidence_count=cited_evidence_count,
            verification_supported=verification.supported,
            latency_ms=total_ms,
        )

        logger.info(
            "RAG timing | total=%.0fms retrieval=%.0fms "
            "generation=%.0fms citation=%.0fms "
            "verification=%.0fms evidence=%d",
            total_ms,
            retrieval_ms,
            generation_ms,
            citation_ms,
            verification_ms,
            len(evidence),
        )

        if not verification.supported:
            answer = GeneratedAnswer(
                answer=ABSTENTION_MESSAGE
            )

        return RAGResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            verification=verification,
            metrics=metrics,
        )