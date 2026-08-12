from langchain_core.tools import tool


@tool
def retrieve(query: str) -> str:
    """检索内部知识库，返回与问题最相关的文档片段。

    当用户的问题涉及项目资料、文档、内部知识库，或需要事实依据时优先使用。
    """
    from rag_agent.retriever import get_retriever

    docs = get_retriever().invoke(query)
    if not docs:
        return "知识库中未找到相关内容。"
    return "\n\n".join(d.page_content for d in docs)


@tool
def calculator(expression: str) -> str:
    """对一段算术表达式求值，例如 "23 * 4 + 1"。当用户需要计算时使用。"""
    try:
        # 仅允许基础算术，避免任意代码执行风险
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "表达式包含不支持的字符。"
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # pragma: no cover
        return f"计算失败：{exc}"


TOOLS = [retrieve, calculator]
