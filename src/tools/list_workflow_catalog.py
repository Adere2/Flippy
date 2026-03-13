import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool

from src.config import get_embeddings

_embed_provider = os.getenv("EMBED_PROVIDER", "google")
CHROMA_DB_DIR = str(
    Path(__file__).resolve().parents[2] / "data" / "chroma_db" / f"workflow_catalog-{_embed_provider}"
)


@tool
def list_workflow_catalog() -> str:
    """
    List all application names available in the Fuzzball Workflow Catalog.
    Use this tool when the user asks what apps, templates, or workflows are available
    in the catalog, or when you want to check whether a specific app might exist before
    doing a full search.
    """
    embeddings = get_embeddings()

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=embeddings
        )

        # Fetch all documents so we can collect every app_name from metadata
        results = vector_store.get(include=["metadatas"])

        if not results or not results.get("metadatas"):
            return "No application templates found in the catalog."

        app_names = sorted(
            {meta["app_name"] for meta in results["metadatas"] if meta.get("app_name")}
        )

        if not app_names:
            return "No application names found in the catalog metadata."

        app_list = "\n".join(f"- {name}" for name in app_names)
        return f"The following applications are available in the Fuzzball Workflow Catalog:\n\n{app_list}"

    except Exception as e:
        return f"Error retrieving app names from the Workflow Catalog: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = list_workflow_catalog_apps.invoke({})
    print(result)
