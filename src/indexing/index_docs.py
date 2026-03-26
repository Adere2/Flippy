import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_embeddings

from src.parsing.hugo_parser import parse_hugo_file

load_dotenv()


def index_fuzzball_docs(docs_dir: str, persist_dir: str):
    """
    Crawls the Hugo docs, parses them, chunks them, and saves them to a Vector DB.
    """
    print(f"🔍 Crawling Fuzzball docs in: {docs_dir}")
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"❌ Error: Directory {docs_dir} does not exist.")
        return

    # 1. Load and Parse all Markdown files
    documents = []
    # rglob("*.md") recursively finds every markdown file in all 88 folders
    for md_file in docs_path.rglob("*.md"):
        doc = parse_hugo_file(md_file)
        if doc:
            documents.append(doc)

    print(f"Successfully parsed {len(documents)} markdown files.")

    # 2. Chunk the Documents
    embed_provider = os.getenv("EMBED_PROVIDER", "google").lower()
    if embed_provider == "ollama":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)

    print("Chunking documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize the Embedding Model
    print(f"Initializing {embed_provider} embedding model...")
    embeddings = get_embeddings()

    # 4. Create and Persist the Vector Database
    print(f"Saving embeddings to Chroma database at {persist_dir}...")

    # This creates the database folder automatically and stores the vectors
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )

    print("Indexing complete! The Knowledge Agent's brain is ready.")


if __name__ == "__main__":
    # Point this to your actual Fuzzball content directory
    SOURCE_DOCS_PATH = os.getenv("FUZZDOCS_DIR")

    if not SOURCE_DOCS_PATH:
        print(
            "❌ Error: FUZZDOCS_DIR environment variable is not set. Please set it in your .env file."
        )
        sys.exit(1)

    # We will store the database right inside our project folder
    embed_provider = os.getenv("EMBED_PROVIDER", "google")
    DATABASE_PATH = f"./data/chroma_db/hugodocs-{embed_provider}"

    index_fuzzball_docs(SOURCE_DOCS_PATH, DATABASE_PATH)
