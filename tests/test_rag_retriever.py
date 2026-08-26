from __future__ import annotations

import unittest
from unittest.mock import patch

from rag_core.config import settings
from rag_core.documents import load_markdown_documents
from rag_core.retriever import RetrievedChunk, keyword_search, retrieve
from rag_core.splitter import split_documents


class _ChunkCollection:
    def __init__(self) -> None:
        self.chunks = split_documents(
            load_markdown_documents(settings.knowledge_base_path),
            settings.chunk_size,
            settings.chunk_overlap,
        )

    def get(self, include: list[str] | None = None) -> dict[str, list[object]]:
        return {
            "ids": [chunk.id for chunk in self.chunks],
            "documents": [chunk.content for chunk in self.chunks],
            "metadatas": [chunk.metadata for chunk in self.chunks],
        }


class KeywordRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = _ChunkCollection()

    def assert_top_match(self, question: str, expected: str) -> None:
        with patch("rag_core.retriever.get_collection", return_value=self.collection):
            results = keyword_search(question, limit=4)
        self.assertTrue(results, question)
        self.assertIn(expected.casefold(), results[0].content.casefold(), question)

    def test_glossary_queries_find_the_expected_entry(self) -> None:
        cases = (
            ("Что такое prompt injection?", "Prompt injection"),
            ("А непрямая промпт-инъекция?", "Indirect prompt injection"),
            ("Что такое Excessive Agency?", "Excessive Agency"),
            ("Кто такие OWASP?", "Open Worldwide Application Security Project"),
            ("Что регулирует 152-ФЗ?", "152-ФЗ"),
            ("Чем псевдонимизация отличается от анонимизации?", "Псевдонимизация"),
            ("Что такое Hugging Face?", "Hugging Face"),
            ("Что такое ExploitGym?", "ExploitGym"),
            ("Что значит zero-day?", "Zero-day"),
            ("Кто такие OpenAI?", "## OpenAI"),
            ("Кто делает Claude?", "## Anthropic"),
            ("Какие есть российские компании по ИИ?", "Российские компании"),
            ("Какие есть китайские нейросети?", "Китайские компании"),
            ("Какая модель сейчас самая сильная?", "Artificial Analysis"),
            ("Какая китайская модель сейчас сильная?", "GLM-5.2"),
            ("Кто такой тревожный пирожок?", "Тревожный пирожок"),
            ("Я тревожный пирожок?", "Тревожный пирожок"),
            ("Кто такой Инспектор Колобок?", "Инспектор Колобок"),
            ("Что расследует Инспектор Колобок?", "Инспектор Колобок"),
            ("Что такое harness?", "Harness"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                self.assert_top_match(question, expected)

    def test_zero_day_hyphen_and_space_are_equivalent(self) -> None:
        with patch("rag_core.retriever.get_collection", return_value=self.collection):
            hyphenated = keyword_search("zero-day", limit=1)
            spaced = keyword_search("zero day", limit=1)
        self.assertEqual(hyphenated[0].id, spaced[0].id)

    def test_hybrid_results_include_keyword_and_semantic_matches(self) -> None:
        semantic = [RetrievedChunk("semantic", "semantic", {}, retrieval_method="semantic")]
        keyword = [RetrievedChunk("keyword", "keyword", {}, retrieval_method="keyword")]
        with (
            patch("rag_core.retriever.semantic_search", return_value=semantic),
            patch("rag_core.retriever.keyword_search", return_value=keyword),
        ):
            results = retrieve("question", top_k=2)
        self.assertEqual([chunk.id for chunk in results], ["keyword", "semantic"])


if __name__ == "__main__":
    unittest.main()
