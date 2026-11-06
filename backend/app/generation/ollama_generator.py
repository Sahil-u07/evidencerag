from dataclasses import dataclass

from app.generation.streaming import stream_generate


@dataclass
class GeneratedAnswer:
    answer: str
    truncated: bool = False


class OllamaGenerator:
    """Non-streaming view of the same Ollama stream that /query/stream uses."""

    def generate(self, query: str, evidence: list) -> GeneratedAnswer:
        tokens = list(stream_generate(query, evidence))

        return GeneratedAnswer(
            answer="".join(token.text for token in tokens).strip(),
            truncated=any(token.truncated for token in tokens),
        )
