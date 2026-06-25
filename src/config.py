"""Settings — single source of truth, loaded from .env. See TDD §9."""
from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cohere_api_key: SecretStr
    embed_model: str = "embed-v4.0"
    chat_model: str = "command-a-03-2025"
    kafka_bootstrap: str = "localhost:9092"
    raw_topic: str = "raw-docs"
    rejected_topic: str = "rejected"
    raw_partitions: int = 4
    hf_slice: str = "HuggingFaceFW/fineweb"  # verify license + sample config (OQ3)
    sample_size: int = 200_000
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    model_path: Path = Path("models/quality_model.json")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # type: ignore[call-arg]
