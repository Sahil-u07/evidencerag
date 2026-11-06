from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.ingestion.models import DocumentChunk


class PersistentIndexStore:
    """Persist document chunks, embeddings, and source fingerprints."""

    VERSION = 3
    INDEX_FILENAME = "index.json"
    EMBEDDINGS_FILENAME = "embeddings.npy"

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.index_path = self.storage_dir / self.INDEX_FILENAME
        self.embeddings_path = self.storage_dir / self.EMBEDDINGS_FILENAME

    def exists(self) -> bool:
        return self.index_path.exists() and self.embeddings_path.exists()

    @staticmethod
    def _validate_embeddings(
        chunks: list[DocumentChunk], embeddings: list[list[float]] | np.ndarray
    ) -> np.ndarray:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        matrix = np.asarray(embeddings, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        if len(chunks) != len(matrix):
            raise ValueError("Number of chunks must match number of embeddings")

        return matrix

    @staticmethod
    def fingerprint_file(path: str | Path) -> str:
        file_path = Path(path)
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()

    @classmethod
    def build_manifest(cls, paths: list[str | Path]) -> dict[str, str]:
        manifest: dict[str, str] = {}

        for path in sorted((Path(path) for path in paths), key=lambda item: str(item)):
            if not path.is_file():
                continue

            manifest[str(path.resolve())] = cls.fingerprint_file(path)

        return manifest

    def save(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | np.ndarray,
        manifest: dict[str, str] | None = None,
    ) -> None:
        """Persist chunks, embeddings, and source fingerprints atomically."""
        matrix = self._validate_embeddings(chunks, embeddings)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": self.VERSION,
            "manifest": manifest or {},
            "chunks": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }

        index_tmp = self.index_path.with_suffix(".tmp")
        embeddings_tmp = self.embeddings_path.with_suffix(".tmp")

        index_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with embeddings_tmp.open("wb") as file:
            np.save(file, matrix)

        index_tmp.replace(self.index_path)
        embeddings_tmp.replace(self.embeddings_path)

    def load(self) -> tuple[list[DocumentChunk], np.ndarray]:
        if not self.exists():
            raise FileNotFoundError("Persistent index does not exist")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))

        if payload.get("version") != self.VERSION:
            raise ValueError("Unsupported persistent index version")

        chunks = [
            DocumentChunk(
                text=item["text"],
                source=item["source"],
                page=item.get("page"),
                chunk_id=item.get("chunk_id"),
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("chunks", [])
        ]

        embeddings = np.load(self.embeddings_path, allow_pickle=False)

        if embeddings.ndim != 2:
            raise ValueError("Persisted embeddings must be a 2D matrix")

        if len(chunks) != len(embeddings):
            raise ValueError("Persisted chunks and embeddings are out of sync")

        return (chunks, embeddings.astype(np.float32, copy=False))

    def load_manifest(self) -> dict[str, str]:
        if not self.exists():
            raise FileNotFoundError("Persistent index does not exist")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))

        if payload.get("version") != self.VERSION:
            raise ValueError("Unsupported persistent index version")

        return payload.get("manifest", {})

    def matches_manifest(self, manifest: dict[str, str]) -> bool:
        if not self.exists():
            return False

        try:
            return self.load_manifest() == manifest
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def clear(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()

        if self.embeddings_path.exists():
            self.embeddings_path.unlink()
