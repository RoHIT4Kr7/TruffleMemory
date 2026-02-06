from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    whatsapp_dir: Path
    processed_dir: Path
    memory_dir: Path
    chroma_dir: Path
    sqlite_path: Path
    profile_path: Path
    openrouter_api_key: str
    openrouter_base_url: str
    model_name: str
    model_temperature: float
    initial_token_budget: int
    recall_token_budget: int
    max_recall_count: int
    chunk_max_chars: int
    chunk_gap_hours: float
    deep_salience_threshold: float
    medium_salience_threshold: float
    max_workers: int


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    data_dir = project_root / "data"
    memory_dir = data_dir / "memory"

    settings = Settings(
        project_root=project_root,
        data_dir=data_dir,
        whatsapp_dir=data_dir / "whatsapp",
        processed_dir=data_dir / "processed",
        memory_dir=memory_dir,
        chroma_dir=memory_dir / "chromadb",
        sqlite_path=memory_dir / "memory.db",
        profile_path=memory_dir / "profile.json",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip(),
        model_name=os.getenv(
            "OPENROUTER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"
        ).strip(),
        model_temperature=_env_float("MODEL_TEMPERATURE", 0.2),
        initial_token_budget=_env_int("INITIAL_TOKEN_BUDGET", 100_000),
        recall_token_budget=_env_int("RECALL_TOKEN_BUDGET", 30_000),
        max_recall_count=_env_int("MAX_RECALL_COUNT", 3),
        chunk_max_chars=_env_int("CHUNK_MAX_CHARS", 2_200),
        chunk_gap_hours=_env_float("CHUNK_GAP_HOURS", 2.0),
        deep_salience_threshold=_env_float("DEEP_SALIENCE_THRESHOLD", 0.65),
        medium_salience_threshold=_env_float("MEDIUM_SALIENCE_THRESHOLD", 0.40),
        max_workers=_env_int("MAX_WORKERS", 6),
    )
    ensure_directories(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.whatsapp_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
