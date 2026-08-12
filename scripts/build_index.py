"""Standalone helper: build the vector index from data/ without running the agent.

Backend is selected by VECTOR_BACKEND (faiss | milvus); SQL docstore is used
when SQL_DOCSTORE_ENABLED=true. See rag_agent/config.py / .env.example.
"""
import sys

from rag_agent.config import settings
from rag_agent.retriever import build_index

if __name__ == "__main__":
    build_index()
    print(f"Built index with backend={settings.VECTOR_BACKEND} "
          f"(sql_docstore={'on' if settings.SQL_DOCSTORE_ENABLED else 'off'})")
    sys.exit(0)
