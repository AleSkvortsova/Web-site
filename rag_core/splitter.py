from __future__ import annotations

from dataclasses import dataclass

from rag_core.documents import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, str | int]


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def split_documents(
    documents: list[LoadedDocument], chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

    chunks: list[DocumentChunk] = []
    for document in documents:
        current = ""
        chunk_index = 0

        def append_chunk(content: str) -> None:
            nonlocal chunk_index
            chunk = content.strip()
            if not chunk:
                return
            chunks.append(
                DocumentChunk(
                    id=f"{document.source}:{chunk_index}",
                    content=chunk,
                    metadata={
                        "source": document.source,
                        "title": document.title,
                        "document_type": document.document_type,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

        for paragraph in _paragraphs(document.content):
            if len(paragraph) > chunk_size:
                if current:
                    append_chunk(current)
                    current = ""
                for piece in _split_long_text(paragraph, chunk_size, chunk_overlap):
                    append_chunk(piece)
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                append_chunk(current)
                overlap = current[-chunk_overlap:].strip() if chunk_overlap else ""
                current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph

        append_chunk(current)

    return chunks

