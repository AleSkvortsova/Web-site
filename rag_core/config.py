from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled at runtime by requirements
    load_dotenv = None


if load_dotenv:
    load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RAGSettings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

    knowledge_base_path: Path = _path_from_env("RAG_KNOWLEDGE_BASE_PATH", "knowledge_base")
    chroma_db_path: Path = _path_from_env("RAG_CHROMA_DB_PATH", "rag_core/chroma_db")
    collection_name: str = os.getenv("RAG_CHROMA_COLLECTION", "site_knowledge_base")

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "140"))
    retriever_top_k: int = int(os.getenv("RAG_RETRIEVER_TOP_K", "4"))
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0"))


settings = RAGSettings()

