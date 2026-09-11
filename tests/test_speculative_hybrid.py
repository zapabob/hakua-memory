"""Unit tests for speculative-hybrid draft/verify retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from hakua_memory.composite import CompositeMemory
from hakua_memory.semantic_graph.embedding.base import EmbeddingModelIdentity
from hakua_memory.semantic_graph.retrieval import (
    lexical_margin,
    speculative_hybrid_search_and_rank,
)


class _CountingBackend:
    """Minimal backend that records embed calls and returns fixed vectors."""

    def __init__(self, dimensions: int = 8) -> None:
        self.identity = EmbeddingModelIdentity(
            provider="test",
            model="counting",
            revision="1",
            dimensions=dimensions,
            serializer_version=1,
        )
        self.embed_query_calls = 0
        self.embed_documents_calls = 0
        self._dimensions = dimensions

    def available(self) -> bool:
        return True

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return [0.1] * self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [[0.1] * self._dimensions for _ in texts]


def _seed_memory(root: Path) -> CompositeMemory:
    memory = CompositeMemory(root, enable_rag=False)
    memory.add_node(
        {
            "node_id": "benchmark-alpha",
            "node_type": "Claim",
            "label": "Release Schedule",
            "summary": "Release is planned for 2026-09-30 with freeze week.",
            "status": "asserted",
            "confidence": 0.95,
            "salience": 0.9,
        }
    )
    memory.add_node(
        {
            "node_id": "benchmark-beta",
            "node_type": "Claim",
            "label": "Budget Approval",
            "summary": "Q3 budget was approved by finance.",
            "status": "asserted",
            "confidence": 0.7,
            "salience": 0.5,
        }
    )
    return memory


def test_lexical_margin_helpers() -> None:
    assert lexical_margin([]) == 0.0
    assert lexical_margin([{"final_score": 0.8}]) == 1.0
    assert lexical_margin([{"final_score": 0.8}, {"final_score": 0.5}]) == pytest.approx(0.3)


def test_speculative_early_accept_skips_dense(tmp_path: Path) -> None:
    memory = _seed_memory(tmp_path / "early")
    backend = _CountingBackend()
    try:
        results = speculative_hybrid_search_and_rank(
            memory.semantic,
            "Release Schedule",
            backend=backend,
            top_k=5,
            margin_threshold=0.0,
            min_top_score=0.0,
        )
        assert results
        assert results[0]["speculative_path"] == "lexical_early_accept"
        assert backend.embed_query_calls == 0
    finally:
        memory.close()


def test_speculative_verify_calls_dense_when_uncertain(tmp_path: Path) -> None:
    memory = _seed_memory(tmp_path / "verify")
    backend = _CountingBackend()
    try:
        results = speculative_hybrid_search_and_rank(
            memory.semantic,
            "Release Schedule",
            backend=backend,
            top_k=5,
            margin_threshold=10.0,
            min_top_score=0.0,
        )
        assert results
        assert results[0]["speculative_path"] == "hybrid_verify"
        assert backend.embed_query_calls >= 1
    finally:
        memory.close()


def test_composite_search_speculative_flag(tmp_path: Path) -> None:
    memory = _seed_memory(tmp_path / "composite")
    backend = _CountingBackend()
    try:
        accepted = memory.search(
            "Release Schedule",
            top_k=5,
            backend=backend,
            speculative=True,
            margin_threshold=0.0,
            min_top_score=0.0,
        )
        assert accepted[0]["speculative_path"] == "lexical_early_accept"
        assert backend.embed_query_calls == 0
    finally:
        memory.close()
