"""Standalone helper: build the FAISS index from data/ without running the agent."""
import sys

from rag_agent.config import settings
from rag_agent.retriever import build_vectorstore

if __name__ == "__main__":
    vs = build_vectorstore()
    print(f"Built index with {vs.index.ntotal} vectors -> {settings.INDEX_PATH}")
    sys.exit(0)
