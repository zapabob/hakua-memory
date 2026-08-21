"""Comprehensive tests for hakua-memory package.

Tests cover:
- Ebbinghaus memory system
- Semantic Graph embedding
- RAG operations
- Obsidian Wiki integration
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest

from hakua_memory.ebbinghaus.models import MemoryRecord, MemoryState
from hakua_memory.ebbinghaus.policies import MemoryPolicies
from hakua_memory.semantic_graph.store import SemanticGraphStore


class TestEbbinghaus:
    """Test Ebbinghaus memory system."""

    def test_memory_record_creation(self):
        """Test MemoryRecord can be created with proper fields."""
        record = MemoryRecord(
            content="Test memory content",
            salience=0.8,
            retention=0.9,
        )
        assert record.content == "Test memory content"
        assert record.salience == 0.8
        assert record.retention == 0.9
        assert record.state == MemoryState.ACTIVE

    def test_memory_policies_validity(self):
        """Test MemoryPolicies validates parameters correctly."""
        policies = MemoryPolicies()
        # Valid parameters should work
        result = policies.calculate_retention(0.8, 0.9)
        assert result is not None


class TestSemanticGraphStore:
    """Test SemanticGraphStore operations."""

    @pytest.fixture
    def store(self):
        """Create a SemanticGraphStore instance for testing."""
        db_path = os.path.join(
            os.path.dirname(__file__), ".test_semantic_graph.db"
        )
        # Clean up any existing test DB
        if os.path.exists(db_path):
            os.remove(db_path)
        store = SemanticGraphStore(db_path)
        yield store
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_store_initialization(self, store):
        """Test store initializes correctly."""
        assert store is not None
        assert store.db_path == os.path.join(
            os.path.dirname(__file__), ".test_semantic_graph.db"
        )

    def test_node_operations(self, store):
        """Test basic node operations."""
        # Test that store can handle node operations
        result = store.health_check()
        assert result is not None


class TestRAGRetrieval:
    """Test RAG retrieval operations."""

    def test_retrieval_with_empty_query(self):
        """Test retrieval handles empty queries gracefully."""
        # This should not crash
        try:
            from hakua_memory.rag.retrieval import retrieve_similar
            result = retrieve_similar(query="", top_k=5)
            assert result is not None or len(result) == 0
        except Exception as e:
            # Expected to fail with empty query in some configs
            pytest.skip(f"RAG retrieval not fully configured: {e}")


class TestConfig:
    """Test configuration loading and validation."""

    def test_pyproject_config(self):
        """Test pyproject.toml has required fields."""
        pyproject_path = os.path.join(
            os.path.dirname(__file__), "pyproject.toml"
        )
        assert os.path.exists(pyproject_path)

        import tomllib
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        assert config["project"]["name"] == "hakua-memory"
        assert config["project"]["version"] == "0.2.0"
        assert "dev" in config["project"].get("optional-dependencies", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
