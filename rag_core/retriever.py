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
    "какая",
    "какие",
    "какой",
    "такое",
    "такой",
    "значит",
    "why",
    "how",
    "what",
    "the",
    "and",
    "for",
    "with",
}


DASHES = str.maketrans({
    "-": " ",
    "‐": " ",
    "‑": " ",
    "–": " ",
    "—": " ",
})


def _normalize_text(value: str) -> str:
    normalized = value.lower().replace("ё", "е").translate(DASHES)
    return " ".join(re.findall(r"[\wа-я]+", normalized, flags=re.IGNORECASE))


def _years(question: str) -> list[str]:
    return re.findall(r"\b(20\d{2})\b", question)


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
    words = _normalize_text(question).split()
    return list(dict.fromkeys(word for word in words if len(word) > 2 and word not in STOPWORDS))


def _keyword_score(question: str, content: str, terms: list[str]) -> float:
    normalized_question = _normalize_text(question)
    normalized_content = _normalize_text(content)
    content_words = set(normalized_content.split())

    score = float(sum(1 for term in terms if term in content_words))
    if normalized_question and normalized_question in normalized_content:
        score += 12

    for left, right in zip(terms, terms[1:]):
        if f"{left} {right}" in normalized_content:
            score += 3

    return score


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
        score = _keyword_score(question, content or "", terms)
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
                    score=score,
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

    for index in range(max(len(keyword), len(semantic))):
        for results in (keyword, semantic):
            if index >= len(results):
                continue
            chunk = results[index]
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            merged.append(chunk)
            if len(merged) >= k:
                return merged
    return merged
