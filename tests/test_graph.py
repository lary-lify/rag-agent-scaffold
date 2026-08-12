import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-compile-check")

from rag_agent.graph import build_graph


def test_graph_compiles():
    """Constructing + compiling the LangGraph must succeed without any LLM call."""
    compiled = build_graph()
    assert compiled is not None
    # sanity: the graph exposes the expected nodes
    assert "agent" in compiled.nodes
    assert "tools" in compiled.nodes
