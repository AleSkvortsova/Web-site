from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_core.config import settings
from rag_core.vector_store import get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None
    retrieval_method: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
        }


STOPWORDS = {
    "что",
    "как",
    "для",
    "или",
    "это",
    "если",
    "где",
    "чем",
    "кто",
    "why",
    "how",
    "what",
    "the",
    "and",
    "for",
    "with",
}


def _years(question: str) -> list[str]:
    return [
        year
        for year in re.findall(r"\b(20(?:0[5-9]|1[0-9]|2[0-5]))\b", question)
    ]


def semantic_search(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    collection = get_collection()
    result = collection.query(
        query_texts=[question],
        n_results=top_k or settings.retriever_top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                id=chunk_id,
                content=content,
                metadata=metadata or {},
                score=distance,
                retrieval_method="semantic",
            )
        )
    return chunks


def _keywords(question: str) -> list[str]:
    words = re.findall(r"[\wа-яА-ЯёЁ]+", question.lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def keyword_search(question: str, limit: int | None = None) -> list[RetrievedChunk]:
    terms = _keywords(question)
    years = _years(question)
    if not terms and not years:
        return []

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    matches: list[RetrievedChunk] = []

    for chunk_id, content, metadata in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        content_lower = (content or "").lower()
        score = sum(1 for term in terms if term in content_lower)
        for year in years:
            if re.search(rf"\b{re.escape(year)}\b", content or ""):
                score += 10
                if (metadata or {}).get("source") == "career.md":
                    score += 10
        if score > 0:
            matches.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=content,
                    metadata=metadata or {},
                    score=float(score),
                    retrieval_method="keyword",
                )
            )

    matches.sort(key=lambda chunk: chunk.score or 0, reverse=True)
    return matches[: limit or settings.retriever_top_k]


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    k = top_k or settings.retriever_top_k
    semantic = semantic_search(question, top_k=k)
    keyword = keyword_search(question, limit=k)

    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in [*semantic, *keyword]:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        merged.append(chunk)
        if len(merged) >= k:
            break
    return merged
