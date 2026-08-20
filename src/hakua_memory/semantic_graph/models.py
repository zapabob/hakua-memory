"""Enums and frozen dataclasses for the semantic-graph plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NODE_TYPES = frozenset(
    {
        "Actor",
        "Entity",
        "Concept",
        "Event",
        "Claim",
        "Preference",
        "Goal",
        "Decision",
        "Procedure",
        "Artifact",
        "Evidence",
        "Evaluation",
        "Temporal",
    }
)

EDGE_TYPES = frozenset(
    {
        "about",
        "mentions",
        "relates_to",
        "supports",
        "contradicts",
        "derived_from",
        "produced_by",
        "evaluated_by",
        "part_of",
        "depends_on",
        "caused_by",
        "temporally_precedes",
        "supersedes",
        "same_as",
        "applies_to",
    }
)

STATUSES = frozenset(
    {
        "candidate",
        "asserted",
        "accepted",
        "rejected",
        "superseded",
    }
)

AUTHORITIES = frozenset(
    {
        "user",
        "assistant",
        "tool",
        "external",
        "system",
        "subagent",
    }
)

STRENGTH_LABELS = {
    "weak": 0.25,
    "medium": 0.50,
    "strong": 0.85,
}

DEFAULT_TOOLS = (
    "semantic_graph_status",
    "semantic_graph_begin_run",
    "semantic_graph_ingest",
    "semantic_graph_submit_fragment",
    "semantic_graph_search",
    "semantic_graph_get",
    "semantic_graph_finalize",
    "semantic_graph_evaluate_output",
    "semantic_graph_feedback",
    "semantic_graph_export",
)


@dataclass(frozen=True)
class GraphRun:
    run_id: str
    objective: str
    scope: str
    status: str
    session_id: str = ""
    turn_id: str = ""
    model: str = ""
    platform: str = ""
    schema_version: int = 1
    created_at: str = ""
    finalized_at: str | None = None
    title: str = ""
    summary_artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    artifact_type: str
    content: str
    title: str = ""
    authority: str = "assistant"
    run_id: str | None = None
    session_id: str = ""
    turn_id: str = ""
    task_id: str = ""
    model: str = ""
    platform: str = ""
    content_hash: str = ""
    truncated: bool = False
    redaction_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    subtype: str
    label: str
    summary: str
    status: str
    authority: str
    confidence: float
    salience: float
    identity_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_label: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    relation_label: str
    strength: float
    confidence: float
    status: str
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
