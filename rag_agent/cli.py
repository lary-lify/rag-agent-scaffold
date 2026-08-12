from rag_agent import graph as graph_mod
from rag_agent.config import settings
from rag_agent.retriever import build_vectorstore, get_retriever


def _ensure_index() -> None:
    try:
        get_retriever()
    except Exception:
        print("[index] 未找到本地向量库，正在构建 ...")
        build_vectorstore()
        print("[index] 构建完成。")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG + Agent 脚手架 CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("build-index", help="根据 data/ 下的文档构建 FAISS 向量库")

    run_p = sub.add_parser("run", help="提问并运行 agent")
    run_p.add_argument("question", nargs="?", default=None, help="问题文本")

    args = parser.parse_args()

    if args.cmd == "build-index":
        build_vectorstore()
        print("index built at", settings.INDEX_PATH)
        return

    question = args.question or input("Question: ").strip()
    _ensure_index()
    answer = graph_mod.run(question)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
