import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.config import get_embeddings

load_dotenv()

# Set paths based on environment variables
FUZZFILE_DIR = os.getenv("FUZZFILE_DIR")

if not FUZZFILE_DIR:
    print(
        "❌ Error: FUZZFILE_DIR environment variable is not set. Please set it in your .env file."
    )
    sys.exit(1)

_embed_provider = os.getenv("EMBED_PROVIDER", "google")
CHROMA_DB_DIR = str(
    Path(__file__).resolve().parents[2] / "data" / "chroma_db" / f"fuzzfiles-{_embed_provider}"
)


def index_fuzzfiles():
    print(f"📂 Loading Fuzzfiles from: {FUZZFILE_DIR}")

    # We use TextLoader to read the raw YAML/.fz text.
    # We load each file as a single document without splitting it.
    loader = DirectoryLoader(
        FUZZFILE_DIR,
        glob="**/*.fz",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} Fuzzfiles.")

    # Using the configured embedding model
    embeddings = get_embeddings()

    print(f"🧠 Generating embeddings and saving to: {CHROMA_DB_DIR}...")
    vector_store = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR
    )
    print("✅ Indexing complete!")


if __name__ == "__main__":
    index_fuzzfiles()
