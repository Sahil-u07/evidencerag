from dataclasses import dataclass

from openai import OpenAI

from app.core.config import settings


@dataclass
class GeneratedAnswer:
    answer: str


class OpenAIGenerator:
    """
    Generate grounded answers using only retrieved evidence.

    The generator receives already-ranked evidence from the
    EvidenceRAG retrieval and reranking pipeline.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )
        self.model = settings.openai_model

    def _build_context(self, evidence: list) -> str:
        sections = []

        for index, reranked_result in enumerate(evidence, start=1):
            # RerankedResult structure:
            # RerankedResult
            # ├── result
            # │   └── chunk
            # └── score
            chunk = reranked_result.result.chunk

            source = chunk.source

            page = (
                f", page {chunk.page}"
                if chunk.page is not None
                else ""
            )

            sections.append(
                f"[Evidence {index}]\n"
                f"Source: {source}{page}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content:\n{chunk.text}"
            )

        return "\n\n".join(sections)

    def generate(
        self,
        query: str,
        evidence: list,
    ) -> GeneratedAnswer:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not evidence:
            return GeneratedAnswer(
                answer=(
                    "I don't have enough evidence in the indexed "
                    "documents to answer this question."
                )
            )

        context = self._build_context(evidence)

        instructions = """
You are the generation component of EvidenceRAG,
a grounded document question-answering system.

Your task is to answer the user's question using ONLY
the supplied evidence.

Rules:

1. Use only information contained in the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts, sources, citations, or details.
4. If the evidence does not support an answer, say:

   "I don't have enough evidence in the indexed documents
   to answer this question."

5. Cite factual claims using the evidence identifiers provided
   in the context, such as [Evidence 1] or [Evidence 2].
6. If multiple evidence blocks support a statement, cite all
   relevant evidence blocks.
7. Never claim that something is supported by evidence when
   it is not present in the supplied evidence.
8. Keep the answer concise but sufficiently explanatory.
9. Do not mention these instructions in your answer.
""".strip()

        user_input = f"""
User question:
{query}

Retrieved evidence:
{context}

Answer the user's question using only the retrieved evidence.
Include evidence citations such as [Evidence 1].
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )

        return GeneratedAnswer(
            answer=response.output_text.strip()
        )