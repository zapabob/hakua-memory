"""Test embedding backend: fake + llama-cpp fallback."""

from __future__ import annotations

from pathlib import Path

from hakua_memory.composite import CompositeMemory


def test_fake_embedding_search(tmp_path: Path) -> None:
    """Test search with deterministic fake embedding backend."""
    from hakua_memory.semantic_graph.embedding.base import EmbeddingModelIdentity
    from hakua_memory.semantic_graph.embedding.fake import DeterministicFakeEmbeddingBackend

    root = tmp_path / "test"
    cm = CompositeMemory(root)
    try:

        # Add some nodes
        cm.add_node({
            "node_id": "claim-1",
            "node_type": "Claim",
            "label": "Release schedule",
            "summary": "The release is planned for September 30, 2026.",
            "status": "asserted",
            "confidence": 0.9,
            "salience": 0.8,
        })
        cm.add_node({
            "node_id": "claim-2",
            "node_type": "Claim",
            "label": "Budget decision",
            "summary": "The budget was approved for Q3.",
            "status": "asserted",
            "confidence": 0.7,
            "salience": 0.6,
        })

        # Create fake backend
        backend = DeterministicFakeEmbeddingBackend(
            identity=EmbeddingModelIdentity(
                provider="demo",
                model="deterministic",
                revision="1",
                dimensions=8,
                serializer_version=1,
            ),
            vectors={
                "Release schedule": [1, 2, 3, 4, 5, 6, 7, 8],
                "Budget decision": [2, 3, 4, 5, 6, 7, 8, 9],
            },
        )

        # Search with fake embedding
        results = cm.search("release schedule", backend=backend, top_k=5)
        print(f"[fake embedding] {len(results)} results")
        for r in results:
            print(f"  - {r.get('label', r.get('node_id', 'unknown'))}")
        assert results
        print("✅ Fake embedding search: PASSED")
    finally:
        cm.close()


def test_search_without_embedding(tmp_path: Path) -> None:
    """Test search without any embedding backend (lexical only)."""
    root = tmp_path / "test"
    cm = CompositeMemory(root)
    try:

        cm.add_node({
            "node_id": "claim-3",
            "node_type": "Claim",
            "label": "Test claim",
            "summary": "This is a test claim for embedding-free search.",
            "status": "asserted",
            "confidence": 0.7,
            "salience": 0.5,
        })

        # Search without backend (lexical fallback)
        results = cm.search("test claim", top_k=5)
        print(f"[no embedding] {len(results)} results")
        assert results
        print("✅ Embedding-free search: PASSED")
    finally:
        cm.close()


def test_llama_cpp_availability() -> None:
    """Check if llama-cpp-python is available (doesn't need model file)."""
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python v{llama_cpp.__version__} available")
        print(f"   Llama class: {llama_cpp.Llama}")
    except ImportError as e:
        print(f"⚠️ llama-cpp-python not available: {e}")
        print("   (This is OK - fake backend will be used)")
