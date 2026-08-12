from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag_agent.graph import build_graph

app = FastAPI(title="RAG Agent API", version="0.1.0")


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=Answer)
def chat(query: Query) -> Answer:
    agent = build_graph()
    result = agent.invoke({"messages": [HumanMessage(content=query.question)]})
    return Answer(answer=result["messages"][-1].content)
