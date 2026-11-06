from dataclasses import dataclass
import json
import re
from typing import Iterator

import requests

from app.core.config import settings
from app.core.text import split_sentences


ABSTENTION_MESSAGE = (
    "I don't have enough evidence in the indexed documents to answer this question."
)


# The model reads this many passages, so only these can back its sentences.
GENERATION_CONTEXT_K = 3


@dataclass
class StreamToken:
    text: str
    # Set on a final empty token when Ollama stopped at the token limit.
    truncated: bool = False


def trim_unfinished_sentence(text: str) -> str:
    sentences = split_sentences(text)

    if len(sentences) > 1 and not re.search(r"[.!?]$", sentences[-1]):
        return text[: text.rfind(sentences[-1])].strip()

    return text


def build_context(evidence: list) -> str:
    sections = []

    for index, reranked_result in enumerate(evidence[:GENERATION_CONTEXT_K], start=1):
        chunk = reranked_result.result.chunk

        page = f", page {chunk.page}" if chunk.page is not None else ""

        sections.append(
            f"[Evidence {index}]\nSource: {chunk.source}{page}\nContent:\n{chunk.text}"
        )

    return "\n\n".join(sections)


def build_prompt(query: str, evidence: list) -> str:
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


def stream_generate(query: str, evidence: list) -> Iterator[StreamToken]:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not evidence:
        yield StreamToken(text=ABSTENTION_MESSAGE)
        return

    prompt = build_prompt(query=query, evidence=evidence)

    response = requests.post(
        settings.generate_url,
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": 2048, "num_predict": 80},
        },
        stream=True,
        timeout=120,
    )

    try:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")

            if token:
                yield StreamToken(text=str(token))

            if data.get("done"):
                if data.get("done_reason") == "length":
                    yield StreamToken(text="", truncated=True)

                break

    finally:
        response.close()


def describe_error(exc: Exception) -> str:
    # ConnectTimeout is both of these, and it means Ollama could not be reached.
    if isinstance(exc, requests.ConnectionError):
        return (
            f"Could not reach Ollama at {settings.ollama_url.rstrip('/')}. "
            "Start Ollama and try again."
        )

    if isinstance(exc, requests.Timeout):
        return (
            "Ollama did not respond in time. "
            "It may still be loading the model; try again."
        )

    if (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code == 404
    ):
        return (
            f"The model {settings.ollama_model} is not available "
            f"in Ollama. Run: ollama pull {settings.ollama_model}"
        )

    return "Something went wrong while answering. Check the backend log."


def generator_available() -> bool:
    model = settings.ollama_model

    if ":" not in model:
        model += ":latest"

    try:
        response = requests.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2
        )
        response.raise_for_status()

        return model in {item["name"] for item in response.json()["models"]}

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return False
