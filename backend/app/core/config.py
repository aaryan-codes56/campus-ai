"""Application settings, read only from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ../.env is the repo-root file when running from backend/. Real environment
    # variables take priority, which is how production is configured.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    # Required on purpose: the app must fail at startup rather than run on a
    # guessed database, and no credential ever lives in the code.
    database_url: str

    # Ingestion
    raw_data_dir: Path = Path("../data/raw")
    manifest_path: Path = Path("../data/manifest.yaml")
    mask_contact_details: bool = True
    max_upload_mb: int = 20
    # Blank disables the admin upload endpoint.
    admin_api_key: str = ""

    # Embeddings
    embedding_provider: Literal["voyage"] = "voyage"
    embedding_model: str = "voyage-4"
    voyage_api_key: str = ""
    # Defaults match an account with no payment method; raise them once one is added.
    voyage_rpm: int = 3
    voyage_tpm: int = 10_000

    # Retrieval. Each is switchable so configurations can be compared in evals.
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    rerank_model: str = "rerank-3"
    retrieval_candidates: int = 30
    rerank_candidates: int = 20
    retrieval_top_k: int = 6
    rrf_k: int = 60
    recency_boost: float = 0.05
    recency_half_life_days: int = 365

    # Answer generation
    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_thinking_level: str = "low"  # blank for models without thinking levels
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "low"
    llm_max_output_tokens: int = 4096
    # Below this best re-rank score, refuse without calling the LLM. Set between measured
    # scores for answerable (0.81-0.88) and unanswerable (0.28-0.47) questions.
    refusal_min_rerank_score: float = 0.55

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
