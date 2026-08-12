from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rag_agent.config import settings
from rag_agent.graph import build_graph

app = FastAPI(title="RAG Agent API", version="0.2.0")

# Build once; the graph carries an in-process checkpointer (MemorySaver) so
# conversations persist across requests for the same session_id (thread_id).
agent = build_graph()


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Gate every chat endpoint. No-op when API_KEY is unset (dev mode)."""
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Query(BaseModel):
    question: str
    session_id: str = "default"


class Answer(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=Answer)
def chat(query: Query, _: None = Depends(verify_api_key)) -> Answer:
    result = agent.invoke(
        {"messages": [HumanMessage(content=query.question)]},
        config={"configurable": {"thread_id": query.session_id}},
    )
    return Answer(answer=result["messages"][-1].content)


@app.post("/api/chat/stream")
async def chat_stream(query: Query, _: None = Depends(verify_api_key)):
    """Server-Sent Events streaming endpoint.

    Yields LLM token chunks as `data: <token>` lines, terminated by
    `data: [DONE]`. Same multi-turn session behavior as /api/chat.
    """
    config = {"configurable": {"thread_id": query.session_id}}

    async def event_generator():
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=query.question)]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
