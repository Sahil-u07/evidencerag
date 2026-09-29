from dataclasses import dataclass
import json
from typing import Iterator

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


@dataclass
class StreamToken:
    text: str


def build_context(evidence: list) -> str:
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


def build_prompt(
    query: str,
    evidence: list,
) -> str:
    context = build_context(evidence)

    return f"""
You are EvidenceRAG's grounded answer generator.

Answer the user's question using ONLY the supplied evidence.

Rules:
- Use only facts explicitly supported by the evidence.
- Do not use outside knowledge.
- Do not invent facts or sources.
- Keep the answer concise and clear.
- Use short sentences.
- Each sentence should contain one main factual claim.
- Do not add citations or evidence IDs.
- If the evidence is insufficient, return exactly:
  I don't have enough evidence in the indexed documents to answer this question.
- Return plain text only.
- Do not use markdown.
- Do not explain these instructions.

USER QUESTION:
{query}

EVIDENCE:
{context}

ANSWER:
""".strip()


def stream_generate(
    query: str,
    evidence: list,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
) -> Iterator[StreamToken]:
    """
    Stream answer text from Ollama token-by-token.

    This intentionally uses plain-text generation rather than
    structured JSON so the frontend can receive answer content
    as soon as Ollama produces it.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not evidence:
        yield StreamToken(
            text=(
                "I don't have enough evidence in the indexed "
                "documents to answer this question."
            )
        )
        return

    prompt = build_prompt(
        query=query,
        evidence=evidence,
    )

    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 96,
            },
        },
        stream=True,
        timeout=120,
    )

    try:
        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True,
        ):
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = data.get(
                "response",
                "",
            )

            if token:
                yield StreamToken(
                    text=str(token),
                )

            if data.get("done"):
                break

    finally:
        response.close()