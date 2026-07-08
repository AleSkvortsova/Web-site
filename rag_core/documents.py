from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    content: str
    source: str
    title: str
    document_type: str = "markdown"


def _title_from_markdown(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def load_markdown_documents(knowledge_base_path: Path) -> list[LoadedDocument]:
    if not knowledge_base_path.exists():
        return []

    documents: list[LoadedDocument] = []
    for path in sorted(knowledge_base_path.glob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append(
            LoadedDocument(
                content=content,
                source=path.name,
                title=_title_from_markdown(content, path.stem),
            )
        )
    return documents

