from dataclasses import dataclass
import json
from typing import Iterator

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the indexed "
    "documents to answer this question."
)


@dataclass
class StreamToken:
    text: str


def build_context(
    evidence: list,
) -> str:
    """
    Build a compact context from the highest-ranked evidence.

    The top-ranked passages are preferred to reduce distraction
    during local-model generation while the API can still return
    the full retrieved source set separately.
    """

    sections = []

    for index, reranked_result in enumerate(
        evidence[:3],
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

IMPORTANT:
- The evidence is ordered from most relevant to least relevant.
- Prefer the highest-ranked relevant evidence.
- If the evidence directly answers the question, answer it directly.
- Do NOT abstain when relevant evidence clearly supports an answer.
- Only abstain when the supplied evidence genuinely does not contain enough information.

Rules:
- Use only facts explicitly supported by the evidence.
- Do not use outside knowledge.
- Do not invent facts or sources.
- Keep the answer concise.
- Use short, clear sentences.
- Each sentence should contain one main factual claim.
- Do not add citations or evidence IDs.
- Do not use markdown.
- Return plain text only.
- If the evidence is insufficient, return exactly:
  {ABSTENTION_MESSAGE}

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
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
    Stream answer text from Ollama as soon as tokens are available.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not evidence:
        yield StreamToken(
            text=ABSTENTION_MESSAGE
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
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 80,
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