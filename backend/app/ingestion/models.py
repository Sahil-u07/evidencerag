from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int | None = None
    chunk_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)