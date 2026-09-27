from dataclasses import dataclass
import re

from app.generation.generator import GeneratedAnswer, OpenAIGenerator
from app.generation.verifier import EvidenceVerifier, VerificationResult


ABSTENTION_MESSAGE = (
    "I don't have enough verified evidence in the indexed "
    "documents to answer this question."
)


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
        Citation normalization
          ↓
        Deterministic citation alignment
          ↓
        Grounding verification
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
        generator=None,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or OpenAIGenerator()
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
        """
        Normalize citation formatting produced by the generator.

        Examples:
            [Evidence1]  -> [Evidence 1]
            [Evidence 2] -> [Evidence 2]
            [ evidence3 ] -> [Evidence 3]
        """

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
        """
        Attach the strongest lexical evidence match to each
        generated sentence.

        Citation assignment is deterministic and happens outside
        the language model so the model does not have to guess
        evidence IDs.
        """

        if not answer.strip() or not evidence:
            return answer

        answer = self._normalize_citations(answer)

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip(),
        )

        cited_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Already cited.
            if self.CITATION_PATTERN.search(sentence):
                cited_sentences.append(
                    self._normalize_citations(sentence)
                )
                continue

            claim_tokens = self._tokens(sentence)

            if not claim_tokens:
                cited_sentences.append(sentence)
                continue

            best_index = None
            best_score = 0.0

            for index, reranked_result in enumerate(
                evidence,
                start=1,
            ):
                evidence_text = (
                    reranked_result.result.chunk.text
                )

                evidence_tokens = self._tokens(
                    evidence_text
                )

                if not evidence_tokens:
                    continue

                overlap = claim_tokens & evidence_tokens

                # Fraction of meaningful claim tokens supported
                # by the evidence.
                score = len(overlap) / len(claim_tokens)

                if score > best_score:
                    best_score = score
                    best_index = index

            # Require substantial lexical support before attaching
            # a citation. This prevents arbitrary citations.
            if (
                best_index is not None
                and best_score >= 0.50
            ):
                sentence = (
                    f"{sentence} [Evidence {best_index}]"
                )

            cited_sentences.append(sentence)

        return " ".join(cited_sentences)

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

        cited_answer = self._attach_citation(
            answer.answer,
            evidence,
        )

        answer = GeneratedAnswer(
            answer=cited_answer,
        )

        verification = self.verifier.verify_grounding(
            answer=answer.answer,
            evidence=evidence,
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
        )