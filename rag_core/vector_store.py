from __future__ import annotations

from typing import Any

from rag_core.config import settings
from rag_core.documents import load_markdown_documents
from rag_core.splitter import split_documents


def _require_openai_key() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before using /api/chat.")


def get_embedding_function() -> Any:
    _require_openai_key()
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Install dependencies from requirements.txt.") from exc

    return OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


def get_chroma_client() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Install dependencies from requirements.txt.") from exc

    settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_db_path))


def get_collection() -> Any:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(),
    )


def index_knowledge_base(reset_collection: bool = False) -> dict[str, int]:
    documents = load_markdown_documents(settings.knowledge_base_path)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    client = get_chroma_client()
    if reset_collection:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(),
    )

    if chunks:
        existing_ids = set(collection.get(include=[]).get("ids", []))
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
        if new_chunks:
            collection.add(
                ids=[chunk.id for chunk in new_chunks],
                documents=[chunk.content for chunk in new_chunks],
                metadatas=[chunk.metadata for chunk in new_chunks],
            )

    return {"documents": len(documents), "chunks": len(chunks)}


def ensure_indexed() -> dict[str, int] | None:
    collection = get_collection()
    if collection.count() > 0:
        return None
    return index_knowledge_base(reset_collection=False)

