from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rag_core.config import settings
from rag_core.generator import generate_answer
from rag_core.retriever import RetrievedChunk, retrieve
from rag_core.vector_store import ensure_indexed


FOLLOW_UP_MARKERS = (
    "этот",
    "эта",
    "это",
    "эти",
    "него",
    "нему",
    "ним",
    "ней",
    "нее",
    "про это",
    "про него",
    "про нее",
    "ссылк",
    "где посмотреть",
    "расскажи подробнее",
    "чем он отличается",
    "чем она отличается",
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        title = chunk.metadata.get("title", "")
        chunk_index = chunk.metadata.get("chunk_index", "")
        parts.append(
            f"[source={source}; title={title}; chunk_index={chunk_index}; id={chunk.id}]\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for chunk in chunks:
        source = chunk.metadata.get("source")
        chunk_index = chunk.metadata.get("chunk_index")
        key = (source, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "title": chunk.metadata.get("title"),
                "chunk_index": chunk_index,
            }
        )
    return sources


def _normalize_history(history: list[dict[str, Any]] | None, limit: int = 6) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content[:1200]})
    return normalized


def _format_history(history: list[dict[str, str]], limit: int = 4) -> str:
    role_names = {
        "user": "Пользователь",
        "assistant": "Фитто",
    }
    lines: list[str] = []
    for item in history[-limit:]:
        role = role_names.get(item["role"], item["role"])
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def _looks_like_follow_up(question: str) -> bool:
    lowered = question.lower().strip()
    return len(lowered) <= 90 or any(marker in lowered for marker in FOLLOW_UP_MARKERS)


def _build_search_query(question: str, history: list[dict[str, str]]) -> str:
    if not history or not _looks_like_follow_up(question):
        return question

    dialogue_context = _format_history(history, limit=2)
    if not dialogue_context:
        return question

    return f"{question}\n\nКонтекст предыдущего диалога:\n{dialogue_context}"


def answer_question(question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question must not be empty.")

    normalized_history = _normalize_history(history)
    search_query = _build_search_query(cleaned_question, normalized_history)
    dialogue_context = _format_history(normalized_history, limit=4)

    index_result = ensure_indexed()
    chunks = retrieve(search_query, top_k=settings.retriever_top_k)
    context = _build_context(chunks)
    answer = generate_answer(cleaned_question, context, dialogue_context=dialogue_context)

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": _build_sources(chunks),
        "retrieved_chunks": [chunk.to_dict() for chunk in chunks],
        "indexed": index_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
