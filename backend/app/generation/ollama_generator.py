from dataclasses import dataclass
import json

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


@dataclass
class GeneratedAnswer:
    answer: str


class OllamaGenerator:
    """
    Local LLM generator using Ollama.

    The model receives only the evidence retrieved by EvidenceRAG
    and returns a structured JSON response containing:
        - answer
        - citations
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model = model
        self.url = url

    def _build_context(
        self,
        evidence: list,
    ) -> str:
        sections = []

        for index, reranked_result in enumerate(
            evidence,
            start=1,
        ):
            chunk = reranked_result.result.chunk

            page = (
                f", page {chunk.page}"
                if chunk.page is not None
                else ""
            )

            sections.append(
                f"[Evidence {index}]\n"
                f"Source: {chunk.source}{page}\n"
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

Answer the user's question using ONLY the supplied evidence.

STRICT RULES:
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent sources.
- Every factual claim must be supported by the supplied evidence.
- Put an evidence citation immediately after every factual claim.
- Use citations exactly like [Evidence 1], [Evidence 2], etc.
- Only cite evidence that supports the claim.
- If the evidence is insufficient, return the exact abstention message.
- Return ONLY valid JSON.
- Do not use markdown code fences.

JSON FORMAT:
{{
  "answer": "Your answer with [Evidence N] citations.",
  "citations": [1]
}}

The citations array must contain every evidence ID used in
the answer.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

RETURN JSON:
""".strip()

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        raw_response = data.get(
            "response",
            "",
        ).strip()

        if not raw_response:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        answer = result.get("answer")
        citations = result.get("citations")

        if not isinstance(answer, str):
            raise RuntimeError(
                "Ollama returned an invalid answer field."
            )

        if not isinstance(citations, list):
            raise RuntimeError(
                "Ollama returned an invalid citations field."
            )

        if not answer.strip():
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        normalized_citations = []

        for citation in citations:
            if not isinstance(citation, int):
                raise RuntimeError(
                    "Ollama returned a non-integer evidence citation."
                )

            if citation < 1 or citation > len(evidence):
                raise RuntimeError(
                    f"Ollama returned invalid evidence citation: "
                    f"{citation}"
                )

            normalized_citations.append(citation)

        normalized_citations = sorted(
            set(normalized_citations)
        )

        # Ensure the structured citations are also visible
        # to EvidenceRAG's grounding verifier.
        for evidence_id in normalized_citations:
            marker = f"[Evidence {evidence_id}]"

            if marker not in answer:
                answer = (
                    answer.rstrip()
                    + f" {marker}"
                )

        return GeneratedAnswer(
            answer=answer.strip()
        )