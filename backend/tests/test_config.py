import os
from pathlib import Path
import subprocess
import sys

from app.core.config import Settings


BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent


def test_defaults(monkeypatch):
    for name in ("OLLAMA_URL", "OLLAMA_MODEL", "MAX_UPLOAD_SIZE_MB", "INDEX_STORAGE_DIR", "UPLOAD_DIR"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.ollama_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2:1b"
    assert settings.max_upload_size_mb == 10
    assert settings.index_storage_dir == "data/index"
    assert settings.upload_dir == "data/uploads"


def test_env_file_is_the_repo_root_one():
    assert Path(Settings.model_config["env_file"]) == REPO_ROOT / ".env"


def test_env_example_lists_every_setting_with_its_default():
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    example = dict(line.split("=", 1) for line in lines if line.strip() and not line.startswith("#"))
    defaults = {name.upper(): str(field.default) for name, field in Settings.model_fields.items()}

    assert example == defaults


def test_app_imports_with_no_env_file_and_no_keys(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("OPENAI_", "OLLAMA_"))}
    env["PYTHONPATH"] = str(BACKEND)

    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
