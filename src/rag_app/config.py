import os
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # pydantic-settings config: read from .env and preserve case
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER")
    OPENAI_API_KEY: str | None = Field(None, env="OPENAI_API_KEY")
    OPENAI_API_BASE: str | None = Field(None, env="OPENAI_API_BASE")
    LLM_MODEL: str = Field("gpt-3.5-turbo", env="LLM_MODEL")
    LOCAL_MODEL_PATH: str | None = Field(None, env="LOCAL_MODEL_PATH")
    EMBEDDING_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    RERANKER_MODEL: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")
    CHROMA_PERSIST_DIR: str = Field("./chromadb", env="CHROMA_PERSIST_DIR")
    CANDIDATE_POOL: int = Field(100, env="CANDIDATE_POOL")
    TOP_K_RERANK: int = Field(10, env="TOP_K_RERANK")
    TOP_K_ANSWER: int = Field(5, env="TOP_K_ANSWER")
    TEMPERATURE: float = Field(0.0, env="TEMPERATURE")
    MAX_TOKENS: int = Field(1024, env="MAX_TOKENS")

    

    @validator("LLM_PROVIDER")
    def normalize_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"openai", "local"}:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'local'")
        return normalized

    @validator("OPENAI_API_KEY", always=True)
    def validate_openai_api_key(cls, value: str | None, values: dict) -> str | None:
        provider = values.get("LLM_PROVIDER")
        if provider == "openai" and not value:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return value

    @validator("LOCAL_MODEL_PATH", always=True)
    def validate_local_model_path(cls, value: str | None, values: dict) -> str | None:
        provider = values.get("LLM_PROVIDER")
        if provider == "local" and not value:
            raise ValueError("LOCAL_MODEL_PATH is required when LLM_PROVIDER=local")
        return value

    @property
    def persist_path(self) -> Path:
        path = Path(self.CHROMA_PERSIST_DIR).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
