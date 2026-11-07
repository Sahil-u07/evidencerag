from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    max_upload_size_mb: int = 10
    index_storage_dir: str = "data/index"
    upload_dir: str = "data/uploads"
    # Public demo: no uploads or deletes, and generation is queued.
    read_only: bool = False
    # Built frontend to serve at "/"; empty keeps the JSON root.
    frontend_dist: str = ""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def generate_url(self) -> str:
        return f"{self.ollama_url.rstrip('/')}/api/generate"


settings = Settings()
