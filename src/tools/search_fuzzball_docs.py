import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import tool

from src.config import get_embeddings

load_dotenv()

# 1. Initialize the database connection
project_root = Path(__file__).resolve().parents[2]
_embed_provider = os.getenv("EMBED_PROVIDER", "google")
db_path = project_root / "data" / "chroma_db" / f"hugodocs-{_embed_provider}"

embeddings = get_embeddings()
vectorstore = Chroma(persist_directory=str(db_path), embedding_function=embeddings)

# Create a retriever that pulls the top 5 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def search_fuzzball_docs(query: str) -> str:
    """
    Use this tool to search the official Fuzzball documentation.
    Pass in a specific, detailed search query.
    Returns excerpts from the documentation to help answer user questions about Fuzzball.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documentation found for that query."

    # Format the results cleanly so the LLM can easily read them
    results = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Title")
        results.append(f"--- Document: {title} ---\n{doc.page_content}\n")

    return "\n".join(results)


if __name__ == "__main__":
    result = search_fuzzball_docs.invoke({"query": "who created fuzzball"})
    print(result)
