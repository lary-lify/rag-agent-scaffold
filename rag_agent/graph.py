from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_agent.llm import get_chat_model
from rag_agent.tools import TOOLS

_model = get_chat_model().bind_tools(TOOLS)


def _agent(state: MessagesState):
    """LLM 节点：根据历史消息（及已绑定工具）决定下一步——直接回答或调用工具。"""
    response = _model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """构建并编译 RAG + Agent 图。

    流程：START -> agent（LLM）-> 若有工具调用则 tools -> agent（循环），
    直到 LLM 不再请求工具，回到 END。
    """
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run(question: str) -> str:
    """便捷封装：用单条问题运行图，返回最终回答文本。"""
    app = build_graph()
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
