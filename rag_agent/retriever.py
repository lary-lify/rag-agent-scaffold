from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import settings
from rag_agent.llm import get_embeddings


# --------------------------------------------------------------------------
# 文档加载与切分（两种后端共用）
# --------------------------------------------------------------------------
def load_raw_documents() -> List[Document]:
    data_dir = Path(settings.DATA_DIR)
    raw: List[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        raw.extend(TextLoader(str(path), encoding="utf-8").load())
    if not raw:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 .txt 文档")
    return raw


def split_documents(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)


# --------------------------------------------------------------------------
# SQL 文档库：把 chunk 文本与来源存到 SQL（MySQL / PostgreSQL / SQLite 均可）
# 推荐与 Milvus 配合——Milvus 存「向量」，这里存「文本」，检索时按 id 关联。
# 依赖（sqlalchemy）延迟导入，默认 FAISS 路径不会触发。
# --------------------------------------------------------------------------
class SQLDocStore:
    def __init__(self, dsn: Optional[str] = None, table: Optional[str] = None):
        from sqlalchemy import (Column, Integer, MetaData, String, Table, Text,
                                create_engine)

        self.dsn = dsn or settings.SQL_DOCSTORE_DSN or "sqlite:///./rag_docs.db"
        self.table_name = table or settings.SQL_DOCSTORE_TABLE
        self.engine = create_engine(self.dsn)
        self.metadata = MetaData()
        self.table = Table(
            self.table_name, self.metadata,
            Column("id", Integer, primary_key=True),
            Column("content", Text, nullable=False),
            Column("source", String(512), default=""),
        )
        self.metadata.create_all(self.engine)

    def save(self, chunks: List[Document]) -> None:
        from sqlalchemy import insert

        rows = [
            {"id": i, "content": d.page_content,
             "source": d.metadata.get("source", "")}
            for i, d in enumerate(chunks)
        ]
        with self.engine.begin() as conn:
            conn.execute(insert(self.table), rows)

    def get(self, ids: List[int]) -> List[Document]:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            rows = conn.execute(
                select(self.table).where(self.table.c.id.in_(ids))
            ).mappings().all()
        by_id = {r["id"]: r for r in rows}
        docs = []
        for i in ids:
            r = by_id.get(i)
            if r:
                docs.append(
                    Document(page_content=r["content"],
                             metadata={"source": r["source"]})
                )
        return docs

    def reset(self) -> None:
        from sqlalchemy import delete

        with self.engine.begin() as conn:
            conn.execute(delete(self.table))


# --------------------------------------------------------------------------
# FAISS 后端（默认）：向量与文本都在本地，零外部服务
# --------------------------------------------------------------------------
def build_faiss_index(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(settings.INDEX_PATH)
    return vectorstore


def load_faiss_retriever():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        settings.INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_TOP_K})


# --------------------------------------------------------------------------
# Milvus 后端：向量在 Milvus，文本在 SQL —— 即「SQL + Milvus」组合
# 依赖（pymilvus）延迟导入，默认路径不会触发。
# --------------------------------------------------------------------------
def _milvus_client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=settings.MILVUS_URI)


def build_milvus_index(chunks: List[Document],
                       docstore: Optional[SQLDocStore] = None):
    from pymilvus import (CollectionSchema, DataType, FieldSchema, MilvusClient)

    embeddings = get_embeddings()
    dim = len(embeddings.embed_query("dimension-probe"))
    client: MilvusClient = _milvus_client()
    collection = settings.MILVUS_COLLECTION

    if client.has_collection(collection):
        client.drop_collection(collection)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
    )
    client.create_collection(collection, schema=schema)

    texts = [d.page_content for d in chunks]
    vectors = embeddings.embed_documents(texts)
    client.insert(
        collection,
        [{"id": i, "vector": vectors[i]} for i in range(len(texts))],
    )

    if docstore is not None:
        docstore.save(chunks)
    return client


class MilvusRetriever:
    """按 query 搜索 Milvus 拿 id，再按需从 SQL 文档库取回文本。"""

    def __init__(self, docstore: SQLDocStore):
        self.client = _milvus_client()
        self.collection = settings.MILVUS_COLLECTION
        self.embeddings = get_embeddings()
        self.docstore = docstore
        self.top_k = settings.RETRIEVER_TOP_K

    def invoke(self, query: str) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection, data=[vector],
            limit=self.top_k, output_fields=[],
        )[0]
        ids = [int(hit["id"]) for hit in hits]
        if not ids:
            return []
        return self.docstore.get(ids)


# --------------------------------------------------------------------------
# 统一入口：根据配置选择后端
# --------------------------------------------------------------------------
def build_index() -> None:
    """根据 VECTOR_BACKEND 构建索引；SQL 文档库按需启用。"""
    chunks = split_documents(load_raw_documents())

    if settings.VECTOR_BACKEND == "milvus":
        if not settings.SQL_DOCSTORE_ENABLED:
            raise ValueError(
                "VECTOR_BACKEND=milvus 必须与 SQL 文档库配合使用："
                "请设置 SQL_DOCSTORE_ENABLED=true（并配置 SQL_DOCSTORE_DSN）。"
            )
        build_milvus_index(chunks, SQLDocStore())
    else:
        # 默认 FAISS；SQL 文档库作为可选辅助存储
        build_faiss_index(chunks)
        if settings.SQL_DOCSTORE_ENABLED:
            SQLDocStore().save(chunks)


def get_retriever():
    """返回统一检索接口（含 .invoke(query) -> list[Document]）。"""
    if settings.VECTOR_BACKEND == "milvus":
        if not settings.SQL_DOCSTORE_ENABLED:
            raise ValueError(
                "VECTOR_BACKEND=milvus 必须与 SQL 文档库配合使用："
                "请设置 SQL_DOCSTORE_ENABLED=true（并配置 SQL_DOCSTORE_DSN）。"
            )
        return MilvusRetriever(SQLDocStore())
    return load_faiss_retriever()
