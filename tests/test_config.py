"""Test configuration management."""

from __future__ import annotations

import tomllib
from pathlib import Path

from hakua_memory.config import (
    EmbeddingConfig,
    HakuaMemoryConfig,
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


def test_config_to_yaml(tmp_path: Path) -> None:
    """Test config serialization to YAML."""
    config = HakuaMemoryConfig()
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    assert path.exists()

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


def test_load_default_config(tmp_path: Path) -> None:
    """Test loading default config (non-existent file returns defaults)."""
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.embedding.backend == "fake"
    print("✅ Load default config: PASSED")


def test_project_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["name"] == "hakua-memory"
    assert project["version"] == "0.3.3"
    assert project["requires-python"] == ">=3.11,<3.14"
    assert set(project["optional-dependencies"]["all"]) == {
        "gitpython>=3,<4",
        "llama-cpp-python>=0.3,<1",
        "pypdf>=4,<5",
        "python-docx>=1,<2",
        "python-pptx>=1,<2",
    }
