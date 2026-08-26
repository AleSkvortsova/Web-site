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


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    def append_section() -> None:
        if heading or body:
            sections.append((heading, "\n\n".join(body).strip()))

    for paragraph in _paragraphs(text):
        if paragraph.startswith("#"):
            append_section()
            heading = paragraph
            body = []
        else:
            body.append(paragraph)
    append_section()
    return sections


def _split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    prefix: str = "",
) -> list[str]:
    chunks: list[str] = []
    available_size = chunk_size - len(prefix) - (2 if prefix else 0)
    if available_size <= chunk_overlap:
        raise ValueError("RAG_CHUNK_SIZE is too small for a markdown heading")
    step = available_size - chunk_overlap
    for start in range(0, len(text), step):
        piece = text[start : start + available_size].strip()
        chunk = f"{prefix}\n\n{piece}".strip() if prefix else piece
        if chunk:
            chunks.append(chunk)
        if start + available_size >= len(text):
            break
    return chunks


def split_documents(
    documents: list[LoadedDocument], chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

    chunks: list[DocumentChunk] = []
    for document in documents:
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

        for heading, body in _sections(document.content):
            section = f"{heading}\n\n{body}".strip() if heading else body
            if len(section) <= chunk_size:
                append_chunk(section)
            else:
                for piece in _split_long_text(body, chunk_size, chunk_overlap, prefix=heading):
                    append_chunk(piece)

    return chunks
