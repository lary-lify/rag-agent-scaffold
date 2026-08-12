"""Document loaders for the RAG pipeline.

Supports plain text (.txt), Markdown (.md) and PDF (.pdf). PDF loading relies
on `pypdf` (added to requirements.txt). Unknown extensions are skipped when
walking a directory.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    # .txt / .md 当作纯文本读取（Markdown 此时不解析为 AST，仅保留原文）
    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


def load_documents(source) -> list[Document]:
    """Load documents from a single file or a directory (walked recursively).

    Args:
        source: path to a file or a directory. Directories are scanned for
            supported extensions only.
    """
    p = Path(source)
    if p.is_file():
        return load_file(p)

    docs: list[Document] = []
    for path in sorted(p.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(path))
    return docs
