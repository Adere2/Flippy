import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import from src
# File is in src/indexing/index_workflow_catalog.py
# parents[0] -> src/indexing
# parents[1] -> src
# parents[2] -> project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from langchain_chroma import Chroma

from src.config import get_embeddings

from src.parsing.workflow_parser import parse_workflow_app

load_dotenv()


def index_workflow_catalog(catalog_dir: str, persist_dir: str):
    """
    Crawls the Workflow Catalog, parses each app directory into a single Document,
    and saves them to a Vector DB. No chunking is used so apps stay intact.
    """
    print(f"🔍 Crawling Workflow Catalog in: {catalog_dir}")
    catalog_path = Path(catalog_dir)

    if not catalog_path.exists():
        print(f"❌ Error: Directory {catalog_dir} does not exist.")
        return

    documents = []

    # Walk through the directories looking for apps
    # os.walk yields (dirpath, dirnames, filenames)
    for root, dirs, files in os.walk(catalog_path):
        if ".git" in root:
            continue

        # Convert string root to Path object
        current_dir = Path(root)

        # Attempt to parse as an app
        doc = parse_workflow_app(current_dir)

        if doc:
            documents.append(doc)

    print(f"✅ Successfully parsed {len(documents)} workflow applications.")

    if not documents:
        print("No documents to index. Exiting.")
        return

    # Using the exact same embedding model used for indexing other collections
    embed_provider = os.getenv("EMBED_PROVIDER", "google")
    print(f"Initializing {embed_provider} embedding model...")
    embeddings = get_embeddings()

    print(f"Saving embeddings to Chroma database at {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=persist_dir
    )

    print("✅ Indexing complete! The Workflow Catalog is ready.")


if __name__ == "__main__":
    # Point this to your actual Workflow Catalog directory via env var
    CATALOG_PATH = os.getenv("WORKFLOW_CATALOG_DIR")

    if not CATALOG_PATH:
        print(
            "❌ Error: WORKFLOW_CATALOG_DIR environment variable is not set. Please set it in your .env file."
        )
        sys.exit(1)

    # We will store the database right inside our project folder
    # Calculating relative to this file: src/indexing/index_workflow_catalog.py -> src/indexing -> src -> project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    embed_provider = os.getenv("EMBED_PROVIDER", "google")
    DATABASE_PATH = str(PROJECT_ROOT / "data" / "chroma_db" / f"workflow_catalog-{embed_provider}")

    index_workflow_catalog(CATALOG_PATH, DATABASE_PATH)
