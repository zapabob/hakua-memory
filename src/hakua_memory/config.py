"""Configuration management for hakua-memory.

Independent from Hermes Agent - manages its own config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding backend configuration.

    Supports:
    - "fake": Deterministic fake backend (default, for tests)
    - "llama-cpp": llama.cpp backend (requires model file)
    - "openai": OpenAI-compatible API
    """

    backend: str = "fake"  # "fake" | "llama-cpp" | "openai"
    dimensions: int = 1024
    serializer_version: int = 1
    timeout_seconds: float = 5.0

    # llama-cpp specific
    model_path: str = ""
    n_ctx: int = 512
    n_gpu_layers: int = -1

    # OpenAI-compatible specific
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        if self.backend not in {"fake", "llama-cpp", "openai"}:
            raise ValueError(f"Unknown embedding backend: {self.backend}")
        if self.dimensions <= 0:
            raise ValueError("embedding.dimensions must be positive")
        if self.backend == "llama-cpp" and not self.model_path:
            raise ValueError("embedding.model_path required for llama-cpp backend")


@dataclass(frozen=True)
class ObsidianConfig:
    """Obsidian export configuration."""

    vault_path: str = ""
    diary_template: str = ""
    dream_template: str = ""
    sleep_log_template: str = ""
    export_diary: bool = True
    export_dream: bool = True
    export_graph: bool = True
    export_forgetting_curve: bool = True

    def __post_init__(self) -> None:
        if self.vault_path and not Path(self.vault_path).exists():
            # Don't raise - vault may be created later
            pass


@dataclass(frozen=True)
class RagConfig:
    """RAG module configuration."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8
    acl_enabled: bool = True
    citation_format: str = "markdown"  # "markdown" | "xml"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("rag.chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("rag.chunk_overlap must be non-negative")


@dataclass(frozen=True)
class HakuaMemoryConfig:
    """Top-level configuration for hakua-memory."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    obsidian: ObsidianConfig = field(default_factory=ObsidianConfig)
    rag: RagConfig = field(default_factory=RagConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> HakuaMemoryConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HakuaMemoryConfig:
        """Create config from dictionary."""
        embedding_data = data.get("embedding", {})
        obsidian_data = data.get("obsidian", {})
        rag_data = data.get("rag", {})

        return cls(
            embedding=EmbeddingConfig(**{k: v for k, v in embedding_data.items() if v is not None}),
            obsidian=ObsidianConfig(**{k: v for k, v in obsidian_data.items() if v is not None}),
            rag=RagConfig(**{k: v for k, v in rag_data.items() if v is not None}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding": {
                "backend": self.embedding.backend,
                "dimensions": self.embedding.dimensions,
                "serializer_version": self.embedding.serializer_version,
                "timeout_seconds": self.embedding.timeout_seconds,
                "model_path": self.embedding.model_path,
                "n_ctx": self.embedding.n_ctx,
                "n_gpu_layers": self.embedding.n_gpu_layers,
                "api_base": self.embedding.api_base,
                "api_key": self.embedding.api_key,
                "model": self.embedding.model,
            },
            "obsidian": {
                "vault_path": self.obsidian.vault_path,
                "diary_template": self.obsidian.diary_template,
                "dream_template": self.obsidian.dream_template,
                "sleep_log_template": self.obsidian.sleep_log_template,
                "export_diary": self.obsidian.export_diary,
                "export_dream": self.obsidian.export_dream,
                "export_graph": self.obsidian.export_graph,
                "export_forgetting_curve": self.obsidian.export_forgetting_curve,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
                "top_k": self.rag.top_k,
                "acl_enabled": self.rag.acl_enabled,
                "citation_format": self.rag.citation_format,
            },
        }

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_default_config_path() -> Path:
    """Get default config file path."""
    return Path.home() / ".hakua-memory" / "config.yaml"


def load_config(path: Optional[Path] = None) -> HakuaMemoryConfig:
    """Load configuration from file, or return defaults."""
    if path is None:
        path = get_default_config_path()
    if path.exists():
        return HakuaMemoryConfig.from_yaml(path)
    return HakuaMemoryConfig()
