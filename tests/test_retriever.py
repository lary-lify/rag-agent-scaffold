"""Tests for the SQL docstore and backend dispatch (no network / API key needed)."""
import pytest
from langchain_core.documents import Document

from rag_agent.retriever import SQLDocStore, build_index, get_retriever


def test_sql_docstore_roundtrip(tmp_path):
    """SQLDocStore must store chunks and return them keyed by id, in requested order."""
    dsn = f"sqlite:///{tmp_path / 'docs.db'}"
    store = SQLDocStore(dsn=dsn, table="chunks")

    chunks = [
        Document(page_content=f"text-{i}", metadata={"source": f"f{i}.txt"})
        for i in range(3)
    ]
    store.save(chunks)

    docs = store.get([2, 0])
    assert len(docs) == 2
    assert docs[0].page_content == "text-2"
    assert docs[1].page_content == "text-0"
    assert docs[0].metadata["source"] == "f2.txt"


def test_milvus_requires_sql_docstore(monkeypatch):
    """VECTOR_BACKEND=milvus without SQL docstore must raise a clear error."""
    from rag_agent import config as config_mod

    monkeypatch.setattr(config_mod.settings, "VECTOR_BACKEND", "milvus")
    monkeypatch.setattr(config_mod.settings, "SQL_DOCSTORE_ENABLED", False)

    with pytest.raises(ValueError):
        get_retriever()
    with pytest.raises(ValueError):
        build_index()
