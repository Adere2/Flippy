import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or "llama3.2",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. Use 'google' or 'ollama'."
        )


def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBED_PROVIDER", "google").lower()
    model = os.getenv("EMBED_MODEL")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model or "nomic-embed-text",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model or "models/gemini-embedding-001"
        )
    else:
        raise ValueError(
            f"Unsupported EMBED_PROVIDER: {provider!r}. Use 'google' or 'ollama'."
        )
