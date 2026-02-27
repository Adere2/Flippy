from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_DB_DIR = str(Path(__file__).resolve().parents[2] / "data/chroma_db/fuzzfiles")


@tool
def search_fuzzfile_examples(query: str) -> str:
    """
    Search for Fuzzfile (.fz) examples and templates based on a user's request.
    Use this tool when you need to see how to write a Fuzzfile for a specific job
    type (e.g., MPI, GPU, simple dependency).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=embeddings
        )

        # Fetch more than we need so we still get 3 unique files after deduplication
        results = vector_store.similarity_search(query, k=6)

        if not results:
            return "No relevant Fuzzfile examples found."

        seen_filenames = set()
        formatted_results = []
        for doc in results:
            # Extract the filename from the metadata to give the LLM context
            source = doc.metadata.get("source", "Unknown File")
            filename = Path(source).name

            # Skip duplicate files — Chroma can return multiple chunks from the same file
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)

            formatted_results.append(
                f"--- Example: {filename} ---\n{doc.page_content}\n"
            )

            # Cap at 3 unique results to avoid blowing up the LLM's context window
            if len(formatted_results) == 3:
                break

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error querying Fuzzfile vector store: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = search_fuzzfile_examples.invoke({"query": "GPU job example"})
    print(result)
