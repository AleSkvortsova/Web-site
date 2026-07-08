from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.config import settings
from rag_core.documents import load_markdown_documents
from rag_core.splitter import split_documents
from rag_core.vector_store import get_chroma_client, get_embedding_function


def main() -> int:
    print("Indexing site knowledge base...")
    print(f"Knowledge base: {settings.knowledge_base_path}")

    documents = load_markdown_documents(settings.knowledge_base_path)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    print(f"Markdown files found: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    if not documents:
        print("No markdown files found. Add files to knowledge_base/ and run again.")
        return 1

    if not chunks:
        print("No chunks created. Check markdown file contents.")
        return 1

    client = get_chroma_client()
    try:
        client.delete_collection(settings.collection_name)
        print(f"Existing collection removed: {settings.collection_name}")
    except Exception:
        print(f"No existing collection to remove: {settings.collection_name}")

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(),
    )
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    print(f"ChromaDB path: {settings.chroma_db_path}")
    print(f"Collection: {settings.collection_name}")
    print("Indexing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

