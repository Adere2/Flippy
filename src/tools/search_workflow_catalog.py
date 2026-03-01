from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Matching your path structure: parents[1] -> src -> parents[2] -> project root
CHROMA_DB_DIR = str(
    Path(__file__).resolve().parents[2] / "data/chroma_db/workflow_catalog"
)


@tool
def search_workflow_catalog(query: str) -> str:
    """
    Search the official Fuzzball Workflow Catalog for application templates.
    Use this tool BEFORE trying to write a complex Fuzzfile from scratch to see if
    an official template already exists for the requested application (e.g., Jupyter, PyTorch).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=embeddings
        )

        # Return only 1 result, the tool "list_workflow_cataglog" improves accuracy.
        results = vector_store.similarity_search(query, k=1)

        if not results:
            return "No relevant application templates found in the catalog."

        formatted_results = []
        for doc in results:
            app_name = doc.metadata.get("app_name", "Unknown App")
            formatted_results.append(
                f"--- App Template: {app_name} ---\n{doc.page_content}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error querying Workflow Catalog vector store: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = search_workflow_catalog.invoke({"query": "specfem3d app earthquake"})
    print(result)
