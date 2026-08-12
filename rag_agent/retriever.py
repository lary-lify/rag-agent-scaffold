from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import settings
from rag_agent.llm import get_embeddings


def build_vectorstore() -> FAISS:
    """Load all *.txt under DATA_DIR, chunk them, embed with FAISS, and persist."""
    embeddings = get_embeddings()
    raw_docs = []
    data_dir = Path(settings.DATA_DIR)
    for path in sorted(data_dir.glob("*.txt")):
        raw_docs.extend(TextLoader(str(path), encoding="utf-8").load())
    if not raw_docs:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(settings.INDEX_PATH)
    return vectorstore


def get_retriever():
    """Lazily load the persisted FAISS index as a retriever."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        settings.INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_TOP_K})
