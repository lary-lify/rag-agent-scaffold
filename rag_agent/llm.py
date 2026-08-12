from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_agent.config import settings


def get_chat_model() -> ChatOpenAI:
    """Chat model used by the agent. Works with any OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Embedding model used to build / load the FAISS index."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_BASE_URL or None,
    )
