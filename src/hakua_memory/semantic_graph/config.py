"""Runtime configuration for the semantic-graph plugin."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .models import DEFAULT_TOOLS

logger = logging.getLogger("hermes.plugins.semantic_graph")

PLUGIN_ID = "semantic-graph"
DB_FILENAME = "semantic_graph.db"
_AUTO_EXTRACT_ALLOWED = frozenset({"off", "explicit", "all"})
_COGNITIVE_MEMORY_MODES = frozenset({"off", "shadow", "active"})

_warn_lock = threading.Lock()
_auto_extract_warned = False


@dataclass(frozen=True)
class SemanticGraphEmbeddingConfig:
    """Typed operator configuration for the single Phase 1 embedding adapter."""

    enabled: bool = False
    backend: str = "llama_cpp"
    endpoint: str = "http://127.0.0.1:8082"
    model: str = "nsfw-bge-m3-v5-q6_k"
    revision: str = ""
    dimensions: int = 1024
    serializer_version: int = 1
    timeout_seconds: float = 5.0
    allow_remote: bool = False

    def __post_init__(self) -> None:
        if self.backend != "llama_cpp":
            raise ValueError("embedding.backend must be llama_cpp")
        if not self.endpoint.strip():
            raise ValueError("embedding.endpoint must not be empty")
        if not self.model.strip():
            raise ValueError("embedding.model must not be empty")
        if isinstance(self.dimensions, bool) or self.dimensions <= 0:
            raise ValueError("embedding.dimensions must be positive")
        if isinstance(self.serializer_version, bool) or self.serializer_version <= 0:
            raise ValueError("embedding.serializer_version must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("embedding.timeout_seconds must be positive")


@dataclass(frozen=True)
class SemanticGraphCognitiveMemoryConfig:
    """Operator-owned cognitive-memory rollout switches."""

    bridge_enabled: bool = False
    rerank_enabled: bool = False
    mode: str = "shadow"
    abstention_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _COGNITIVE_MEMORY_MODES:
            raise ValueError(
                "cognitive_memory.mode must be off, shadow, or active"
            )


@dataclass(frozen=True)
class SemanticGraphConfig:
    db_subdir: str = "semantic-graph"
    capture_turns: bool = True
    capture_tool_events: bool = False
    capture_subagents: bool = True
    auto_extract: str = "explicit"
    retrieval_enabled: bool = True
    retrieval_top_k: int = 8
    retrieval_max_chars: int = 3500
    min_recall_confidence: float = 0.60
    max_artifact_chars: int = 12000
    tool_result_preview_chars: int = 1000
    retention_days: int = 365
    recall_statuses: tuple[str, ...] = ("asserted", "accepted")
    full_tool_result_allowlist: frozenset[str] = field(default_factory=frozenset)
    tool_capture_denylist: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_TOOLS)
    )
    embedding: SemanticGraphEmbeddingConfig = field(
        default_factory=SemanticGraphEmbeddingConfig
    )
    cognitive_memory: SemanticGraphCognitiveMemoryConfig = field(
        default_factory=SemanticGraphCognitiveMemoryConfig
    )

    def db_path(self) -> Path:
        return get_hermes_home() / self.db_subdir / DB_FILENAME

    def export_root(self) -> Path:
        return get_hermes_home() / self.db_subdir / "exports"


def _warn_auto_extract_once(raw: str) -> None:
    global _auto_extract_warned
    with _warn_lock:
        if _auto_extract_warned:
            return
        _auto_extract_warned = True
    logger.warning(
        "semantic-graph: unknown auto_extract=%r; falling back to 'explicit'",
        raw,
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _embedding_positive_int(raw: dict[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"embedding.{key} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedding.{key} must be positive") from exc
    if result <= 0:
        raise ValueError(f"embedding.{key} must be positive")
    return result


def _embedding_positive_float(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"embedding.{key} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedding.{key} must be positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"embedding.{key} must be positive")
    return result


def _raw_plugin_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(PLUGIN_ID) or entries.get("semantic_graph") or {}
        if isinstance(entry, dict):
            nested = entry.get("config")
            return dict(nested) if isinstance(nested, dict) else dict(entry)
    except Exception:
        logger.debug("semantic-graph: config load failed; using defaults", exc_info=True)
    return {}


def load_config(overrides: dict[str, Any] | None = None) -> SemanticGraphConfig:
    raw = _raw_plugin_config()
    if overrides:
        raw = {**raw, **overrides}

    auto_raw = str(raw.get("auto_extract", "explicit") or "explicit").strip().lower()
    if auto_raw not in _AUTO_EXTRACT_ALLOWED:
        _warn_auto_extract_once(auto_raw)
        auto_raw = "explicit"

    recall = raw.get("recall_statuses") or ["asserted", "accepted"]
    if not isinstance(recall, (list, tuple)):
        recall = ["asserted", "accepted"]

    denylist = raw.get("tool_capture_denylist")
    if not isinstance(denylist, (list, tuple)) or not denylist:
        denylist = list(DEFAULT_TOOLS)
    # Always deny own tools to prevent recursive capture.
    denylist = set(str(x) for x in denylist) | set(DEFAULT_TOOLS)

    allowlist = raw.get("full_tool_result_allowlist") or []
    if not isinstance(allowlist, (list, tuple)):
        allowlist = []

    embedding_raw = raw.get("embedding") or {}
    if not isinstance(embedding_raw, dict):
        raise ValueError("embedding must be a mapping")
    embedding = SemanticGraphEmbeddingConfig(
        enabled=_coerce_bool(embedding_raw.get("enabled"), False),
        backend=str(embedding_raw.get("backend") or "llama_cpp").strip(),
        endpoint=str(
            embedding_raw.get("endpoint") or "http://127.0.0.1:8082"
        ).strip(),
        model=str(embedding_raw.get("model") or "nsfw-bge-m3-v5-q6_k").strip(),
        revision=str(embedding_raw.get("revision") or "").strip(),
        dimensions=_embedding_positive_int(embedding_raw, "dimensions", 1024),
        serializer_version=_embedding_positive_int(
            embedding_raw,
            "serializer_version",
            1,
        ),
        timeout_seconds=_embedding_positive_float(
            embedding_raw,
            "timeout_seconds",
            5.0,
        ),
        allow_remote=_coerce_bool(embedding_raw.get("allow_remote"), False),
    )

    cognitive_raw = raw.get("cognitive_memory") or {}
    if not isinstance(cognitive_raw, dict):
        raise ValueError("cognitive_memory must be a mapping")
    cognitive_mode = str(cognitive_raw.get("mode") or "shadow").strip().lower()
    cognitive_memory = SemanticGraphCognitiveMemoryConfig(
        bridge_enabled=_coerce_bool(
            cognitive_raw.get("bridge_enabled"),
            False,
        ),
        rerank_enabled=_coerce_bool(
            cognitive_raw.get("rerank_enabled"),
            False,
        ),
        mode=cognitive_mode,
        abstention_enabled=_coerce_bool(
            cognitive_raw.get("abstention_enabled"),
            False,
        ),
    )

    return SemanticGraphConfig(
        db_subdir=str(raw.get("db_subdir") or "semantic-graph"),
        capture_turns=_coerce_bool(raw.get("capture_turns"), True),
        capture_tool_events=_coerce_bool(raw.get("capture_tool_events"), False),
        capture_subagents=_coerce_bool(raw.get("capture_subagents"), True),
        auto_extract=auto_raw,
        retrieval_enabled=_coerce_bool(raw.get("retrieval_enabled"), True),
        retrieval_top_k=max(1, min(20, _coerce_int(raw.get("retrieval_top_k"), 8))),
        retrieval_max_chars=max(200, _coerce_int(raw.get("retrieval_max_chars"), 3500)),
        min_recall_confidence=max(
            0.0, min(1.0, _coerce_float(raw.get("min_recall_confidence"), 0.60))
        ),
        max_artifact_chars=max(500, _coerce_int(raw.get("max_artifact_chars"), 12000)),
        tool_result_preview_chars=max(
            64, _coerce_int(raw.get("tool_result_preview_chars"), 1000)
        ),
        retention_days=max(0, _coerce_int(raw.get("retention_days"), 365)),
        recall_statuses=tuple(str(x) for x in recall),
        full_tool_result_allowlist=frozenset(str(x) for x in allowlist),
        tool_capture_denylist=frozenset(denylist),
        embedding=embedding,
        cognitive_memory=cognitive_memory,
    )
