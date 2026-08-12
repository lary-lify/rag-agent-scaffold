from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM / Embeddings (OpenAI-compatible, e.g. DeepSeek) ----
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ---- RAG ----
    DATA_DIR: str = "./data"
    INDEX_PATH: str = "./index"
    RETRIEVER_TOP_K: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
