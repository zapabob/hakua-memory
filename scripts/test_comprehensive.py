"""Comprehensive test suite for hakua-memory.

Tests all modules WITHOUT embedding backend (lexical/fake only).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hakua_memory.composite import CompositeMemory
from hakua_memory.config import HakuaMemoryConfig


def test_ebbinghaus_without_embedding() -> None:
    """Test Ebbinghaus memory remember/recall."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        # Remember
        cm.remember("User prefers concise Japanese reports.", tags=["user-pref"])
        cm.remember("Release date is 2026-09-30.", tags=["schedule"])

        # Recall
        results = cm.recall("Japanese reports")
        assert len(results) >= 1
        print(f"✅ Ebbinghaus recall: {len(results)} results")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_semantic_graph_without_embedding() -> None:
    """Test Semantic Graph add_node/search."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        # Add nodes
        cm.add_node({
            "node_id": "claim-1",
            "node_type": "Claim",
            "label": "Release Schedule",
            "summary": "Release is planned for 2026-09-30.",
            "status": "asserted",
            "confidence": 0.9,
            "salience": 0.8,
        })
        cm.add_node({
            "node_id": "claim-2",
            "node_type": "Claim",
            "label": "Budget Approval",
            "summary": "Q3 budget was approved.",
            "status": "asserted",
            "confidence": 0.7,
            "salience": 0.6,
        })

        # Search (no embedding)
        results = cm.search("release", top_k=5)
        assert len(results) >= 1
        print(f"✅ Semantic graph search: {len(results)} results")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rag_without_embedding() -> None:
    """Test RAG ingest/search/citations without embedding."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        # Ingest markdown
        result = cm.ingest_markdown(
            "# Meeting Notes\n決定: Release is 2026-09-30.\nタスク: Alice creates design.",
            title="Meeting 2026-08-20",
            author="Alice",
            department="Engineering",
        )
        doc_id = result["document_id"]
        assert result["chunks"] > 0

        # Search
        results = cm.search_documents("release", top_k=5)
        assert len(results) >= 1

        # Render citations
        context = cm.render_citations(results, format="markdown")
        assert "rag_context" in context
        assert "data_only" in context

        # Extract meeting items
        items = cm.extract_meeting_items(doc_id, auto_store=True)
        assert len(items) >= 1

        # ACL
        cm.grant_access(doc_id, "alice", "read")
        perms = cm.check_access(doc_id, "alice")
        assert perms["can_read"] is True

        print(f"✅ RAG: {len(results)} search, {len(items)} items, ACL ok")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_config_save_load() -> None:
    """Test config YAML save/load."""
    tmpdir = tempfile.mkdtemp()
    try:
        config = HakuaMemoryConfig()
        path = Path(tmpdir) / "config.yaml"
        config.to_yaml(path)

        loaded = HakuaMemoryConfig.from_yaml(path)
        assert loaded.embedding.backend == "fake"
        assert loaded.rag.chunk_size == 1000
        print("✅ Config save/load: PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_obsidian_templates() -> None:
    """Test Obsidian template rendering."""
    from hakua_memory.obsidian.templates import ObsidianTemplateRenderer

    renderer = ObsidianTemplateRenderer()

    # Sleep log
    log = renderer.render_sleep_log(
        review_count=100,
        rehearsal_count=20,
        forget_count=5,
        rehearsal_list=["Memory A"],
        forgotten_list=["Memory B"],
        insights=["Insight 1"],
    )
    assert "レビュー対象: 100件" in log

    # Graph summary
    graph = renderer.render_graph_summary(
        node_count=10,
        active_count=5,
        candidate_count=3,
        rejected_count=2,
        top_nodes=[{"node_id": "n1", "label": "Test", "status": "accepted"}],
    )
    assert "Test" in graph

    print("✅ Obsidian templates: PASSED")


def test_stats() -> None:
    """Test stats aggregation."""
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test"
        cm = CompositeMemory(root)

        cm.remember("Test memory", tags=["test"])
        cm.add_node({
            "node_id": "n1",
            "node_type": "Claim",
            "label": "Test",
            "summary": "Test",
            "status": "asserted",
        })
        cm.ingest_markdown("# Test\nContent", title="Test")

        stats = cm.stats()
        assert "ebbinghaus" in stats
        assert "semantic_graph" in stats
        assert "rag" in stats
        assert stats["rag"]["documents"] >= 1

        print(f"✅ Stats: ebbinghaus={stats['ebbinghaus']['count']}, rag={stats['rag']['documents']}")
    finally:
        cm.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_ebbinghaus_without_embedding()
    test_semantic_graph_without_embedding()
    test_rag_without_embedding()
    test_config_save_load()
    test_obsidian_templates()
    test_stats()
    print("\n★ ALL COMPREHENSIVE TESTS PASSED ★")
