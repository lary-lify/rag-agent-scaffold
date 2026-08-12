from rag_agent import graph as graph_mod
from rag_agent.config import settings
from rag_agent.retriever import build_index, get_retriever


def _ensure_index() -> None:
    try:
        get_retriever()
    except Exception:
        print("[index] 未找到本地向量库，正在构建 ...")
        build_index()
        print("[index] 构建完成。")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG + Agent 脚手架 CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser(
        "build-index",
        help="根据 data/ 下的文档构建向量库（后端由 VECTOR_BACKEND 决定：faiss 或 milvus）",
    )

    run_p = sub.add_parser("run", help="提问并运行 agent")
    run_p.add_argument("question", nargs="?", default=None, help="问题文本")

    args = parser.parse_args()

    if args.cmd == "build-index":
        build_index()
        backend = settings.VECTOR_BACKEND
        print(f"index built with backend={backend} "
              f"(index={settings.INDEX_PATH}, "
              f"sql_docstore={'on' if settings.SQL_DOCSTORE_ENABLED else 'off'})")
        return

    question = args.question or input("Question: ").strip()
    _ensure_index()
    answer = graph_mod.run(question)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
