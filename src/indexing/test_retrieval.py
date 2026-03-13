import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.config import get_embeddings

# Load your Google API key
load_dotenv()


def test_retrieval(query: str):
    print(f"🤔 Asking the Fuzzball brain: '{query}'\n")

    # Calculate the path to the database robustly based on this file's location.
    # __file__ is src/indexing/test_retrieval.py
    # .parents[2] goes up to the project root (fuzzball-assistant/)
    project_root = Path(__file__).resolve().parents[2]
    embed_provider = os.getenv("EMBED_PROVIDER", "google")
    db_path = project_root / "data" / "chroma_db" / f"workflow_catalog-{embed_provider}"

    print(f"📂 Looking for database at: {db_path}")

    # 1. Initialize the EXACT same embedding model used for indexing
    embeddings = get_embeddings()

    # 2. Connect to the local vector database
    vectorstore = Chroma(persist_directory=str(db_path), embedding_function=embeddings)

    # 3. Perform the search (k=3 means return the top 3 most relevant chunks)
    # In Chroma's default settings, a LOWER score means a CLOSER match (it measures distance).
    results = vectorstore.similarity_search_with_score(query, k=3)

    if not results:
        print("❌ No results found. Did the indexing script run successfully?")
        return

    # 4. Print the results clearly
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Result {i} (Distance Score: {score:.4f}) ---")
        # Notice how we can pull the metadata out that your Hugo parser saved!
        print(f"📄 Document Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"🔗 File Path: {doc.metadata.get('source', 'Unknown')}")
        print(f"📝 Excerpt:\n{doc.page_content[:300]}...\n")


if __name__ == "__main__":
    # I picked a question specifically from the markdown snippet you shared earlier!
    test_question = "ai"
    test_retrieval(test_question)
