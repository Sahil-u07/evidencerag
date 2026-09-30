from dataclasses import dataclass
import json

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"


@dataclass
class GeneratedAnswer:
    answer: str


class OllamaGenerator:
    """
    Local LLM generator using Ollama.

    EvidenceRAG supplies only retrieved evidence.
    Citation assignment and grounding verification are
    handled separately by the pipeline.
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
You are EvidenceRAG's grounded answer generator.

Answer the user's question using ONLY the supplied evidence.

Rules:
- Use only facts explicitly supported by the evidence.
- Do not use outside knowledge.
- Do not invent facts or sources.
- Keep the answer concise.
- Use short, clear sentences.
- Each sentence should contain one main factual claim.
- Do not add citations or evidence IDs.
- If the evidence is insufficient, return exactly:
  I don't have enough evidence in the indexed documents to answer this question.
- Return ONLY valid JSON.
- Do not use markdown.

JSON:
{{
  "answer": "Your concise answer."
}}

USER QUESTION:
{query}

EVIDENCE:
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
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 2048,
                    "num_predict": 96,
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

        if not isinstance(answer, str):
            raise RuntimeError(
                "Ollama returned an invalid answer field."
            )

        if not answer.strip():
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        return GeneratedAnswer(
            answer=answer.strip()
        )