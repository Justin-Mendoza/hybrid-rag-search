from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables."""

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql://hybrid_rag:local-development-only@localhost:5432/hybrid_rag"
    redis_url: str = "redis://localhost:6379/0"
    opensearch_url: str = "http://localhost:9200"
    storage_root: Path = Path(".data/originals")
    tokenizer_root: Path = Path(".data/tokenizers")
    cohere_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("COHERE_API_KEY", "COHERE_TRIAL_KEY"),
        repr=False,
    )
    cohere_embed_model: str = "embed-english-light-v3.0"
    cohere_embed_dimensions: int = Field(default=384, gt=0)
    cohere_timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    cohere_embed_usd_per_million_tokens: Decimal | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_rerank_timeout_seconds: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    cohere_rerank_usd_per_search_unit: Decimal | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    cohere_generation_model: str = "command-r7b-12-2024"
    cohere_generation_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    cohere_generation_temperature: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    cohere_generation_input_usd_per_million_tokens: Decimal = Field(
        default=Decimal("0.0375"), ge=0, allow_inf_nan=False
    )
    cohere_generation_output_usd_per_million_tokens: Decimal = Field(
        default=Decimal("0.15"), ge=0, allow_inf_nan=False
    )
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
