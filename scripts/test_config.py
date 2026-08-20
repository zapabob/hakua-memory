"""Test configuration management."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hakua_memory.config import (
    EmbeddingConfig,
    HakuaMemoryConfig,
    ObsidianConfig,
    RagConfig,
    load_config,
)


def test_default_config() -> None:
    """Test default configuration."""
    config = HakuaMemoryConfig()
    assert config.embedding.backend == "fake"
    assert config.embedding.dimensions == 1024
    assert config.rag.chunk_size == 1000
    assert config.obsidian.export_diary is True
    print("✅ Default config: PASSED")


def test_config_to_yaml() -> None:
    """Test config serialization to YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = HakuaMemoryConfig()
        path = Path(tmpdir) / "config.yaml"
        config.to_yaml(path)

        # Verify file was created
        assert path.exists()

        # Load back
        loaded = HakuaMemoryConfig.from_yaml(path)
        assert loaded.embedding.backend == "fake"
        assert loaded.embedding.dimensions == 1024
        print("✅ Config to YAML: PASSED")


def test_config_from_dict() -> None:
    """Test config creation from dictionary."""
    data = {
        "embedding": {
            "backend": "llama-cpp",
            "dimensions": 4096,
            "model_path": "models/llama.gguf",
        },
        "obsidian": {
            "vault_path": "~/obsidian-vault",
            "export_diary": True,
        },
        "rag": {
            "chunk_size": 2000,
            "top_k": 10,
        },
    }
    config = HakuaMemoryConfig.from_dict(data)
    assert config.embedding.backend == "llama-cpp"
    assert config.embedding.dimensions == 4096
    assert config.embedding.model_path == "models/llama.gguf"
    assert config.obsidian.vault_path == "~/obsidian-vault"
    assert config.rag.chunk_size == 2000
    print("✅ Config from dict: PASSED")


def test_config_validation() -> None:
    """Test config validation."""
    # Invalid embedding backend
    try:
        EmbeddingConfig(backend="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown embedding backend" in str(e)

    # Invalid dimensions
    try:
        EmbeddingConfig(dimensions=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positive" in str(e)

    # llama-cpp requires model_path
    try:
        EmbeddingConfig(backend="llama-cpp", model_path="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "model_path" in str(e)

    print("✅ Config validation: PASSED")


def test_load_default_config() -> None:
    """Test loading default config (non-existent file returns defaults)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = load_config(Path(tmpdir) / "nonexistent.yaml")
        assert config.embedding.backend == "fake"
        print("✅ Load default config: PASSED")


if __name__ == "__main__":
    test_default_config()
    test_config_to_yaml()
    test_config_from_dict()
    test_config_validation()
    test_load_default_config()
    print("\n★ ALL CONFIG TESTS PASSED ★")
