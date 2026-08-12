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

    # ---- 向量库后端：faiss（默认，本地零依赖）或 milvus（生产化）----
    VECTOR_BACKEND: str = "faiss"            # faiss | milvus
    MILVUS_URI: str = "./milvus.db"          # Milvus Lite 本地文件，或 "http://host:19530"
    MILVUS_COLLECTION: str = "rag_chunks"

    # ---- SQL 文档库：与 Milvus 配合（向量在 Milvus，文本/元数据在 SQL）----
    SQL_DOCSTORE_ENABLED: bool = False
    SQL_DOCSTORE_DSN: str = ""               # 如 mysql+pymysql://u:p@host:3306/rag；留空用 sqlite
    SQL_DOCSTORE_TABLE: str = "rag_chunks"

    # ---- 服务鉴权 ----
    # 留空则关闭鉴权（仅开发期）。生产请设置强随机值，并在请求头带 X-API-Key。
    API_KEY: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
