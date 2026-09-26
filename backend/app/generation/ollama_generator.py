from dataclasses import dataclass

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


@dataclass
class GeneratedAnswer:
    answer: str


class OllamaGenerator:
    """
    Local LLM generator using Ollama.

    The model receives only the evidence retrieved by EvidenceRAG.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model = model
        self.url = url

    def _build_context(self, evidence: list) -> str:
        sections = []

        for index, reranked_result in enumerate(evidence, start=1):
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

        prompt = f"""
You are the answer-generation component of EvidenceRAG.

Answer the user's question using ONLY the evidence provided below.

STRICT RULES:
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent sources.
- Do not explain your reasoning.
- Give only the final answer.
- Keep the answer concise and clear.
- Cite factual claims using [Evidence 1], [Evidence 2], etc.
- If the evidence does not contain enough information, say exactly:
  "I don't have enough evidence in the indexed documents to answer this question."

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

FINAL ANSWER:
""".strip()

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return GeneratedAnswer(answer=answer)