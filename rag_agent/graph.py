
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_agent.llm import get_chat_model
from rag_agent.tools import TOOLS

# In-process checkpointer: keeps per-session conversation history keyed by
# thread_id. Swap for a persistent checkpointer (e.g. RedisSaver) in production
# so sessions survive process restarts.
_checkpointer = MemorySaver()


def _get_model():
    """延迟构造 LLM：仅在图真正运行时才需要 API key，导入期不触碰凭据。"""
    return get_chat_model().bind_tools(TOOLS)


def _agent(state: MessagesState):
    """LLM 节点：根据历史消息（及已绑定工具）决定下一步——直接回答或调用工具。"""
    response = _get_model().invoke(state["messages"])
    return {"messages": [response]}


def build_graph(checkpointer=None):
    """构建并编译 RAG + Agent 图。

    流程：START -> agent（LLM）-> 若有工具调用则 tools -> agent（循环），
    直到 LLM 不再请求工具，回到 END。
    """
    cp = checkpointer or _checkpointer
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=cp)


def run(question: str, session_id: str = "default") -> str:
    """便捷封装：用单条问题运行图，返回最终回答文本。

    session_id 相同则自动带上历史消息，实现多轮对话。
    """
    app = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    return result["messages"][-1].content
