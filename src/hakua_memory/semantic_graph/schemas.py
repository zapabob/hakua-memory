"""Tool and structured-output JSON schemas for semantic-graph."""

from __future__ import annotations

from typing import Any

from .models import AUTHORITIES, EDGE_TYPES, NODE_TYPES, STATUSES, STRENGTH_LABELS


def _sorted(values) -> list[str]:
    return sorted(values)


GRAPH_FRAGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "nodes", "edges"],
    "properties": {
        "summary": {"type": "string", "maxLength": 4000},
        "nodes": {
            "type": "array",
            "maxItems": 200,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "temp_id",
                    "node_type",
                    "label",
                    "summary",
                    "status",
                    "authority",
                    "confidence",
                    "salience",
                    "evidence",
                ],
                "properties": {
                    "temp_id": {
                        "type": "string",
                        "pattern": "^[A-Za-z0-9_-]{1,64}$",
                    },
                    "node_type": {"type": "string", "enum": _sorted(NODE_TYPES)},
                    "subtype": {"type": "string", "maxLength": 128},
                    "label": {"type": "string", "minLength": 1, "maxLength": 500},
                    "summary": {"type": "string", "maxLength": 4000},
                    "identity_key": {"type": "string", "maxLength": 500},
                    "status": {
                        "type": "string",
                        "enum": [
                            "candidate",
                            "asserted",
                            "accepted",
                            "rejected",
                            "superseded",
                        ],
                    },
                    "authority": {"type": "string", "enum": _sorted(AUTHORITIES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "salience": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "artifact_id",
                                "start_char",
                                "end_char",
                                "quote",
                                "relation",
                                "confidence",
                            ],
                            "properties": {
                                "artifact_id": {"type": "string"},
                                "start_char": {"type": "integer", "minimum": 0},
                                "end_char": {"type": "integer", "minimum": 0},
                                "quote": {"type": "string", "maxLength": 1000},
                                "relation": {
                                    "type": "string",
                                    "enum": [
                                        "supports",
                                        "contradicts",
                                        "mentions",
                                        "derived_from",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                    "metadata": {"type": "object"},
                },
            },
        },
        "edges": {
            "type": "array",
            "maxItems": 400,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_temp_id",
                    "target_temp_id",
                    "edge_type",
                    "strength",
                    "confidence",
                    "status",
                    "rationale",
                    "evidence",
                ],
                "properties": {
                    "source_temp_id": {"type": "string"},
                    "target_temp_id": {"type": "string"},
                    "edge_type": {"type": "string", "enum": _sorted(EDGE_TYPES)},
                    "relation_label": {"type": "string", "maxLength": 128},
                    "strength": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": list(STRENGTH_LABELS.keys()),
                            },
                            {"type": "number", "minimum": 0, "maximum": 1},
                        ]
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "status": {
                        "type": "string",
                        "enum": [
                            "candidate",
                            "asserted",
                            "accepted",
                            "rejected",
                            "superseded",
                        ],
                    },
                    "rationale": {"type": "string", "maxLength": 2000},
                    "evidence": {"type": "array", "maxItems": 20},
                    "metadata": {"type": "object"},
                },
            },
        },
        "evaluations": {
            "type": "array",
            "maxItems": 100,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target_temp_id",
                    "verdict",
                    "score",
                    "criteria",
                    "notes",
                ],
                "properties": {
                    "target_temp_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["support", "uncertain", "revise", "reject"],
                    },
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "criteria": {"type": "object"},
                    "notes": {"type": "string", "maxLength": 4000},
                },
            },
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 100,
            "items": {"type": "string", "maxLength": 1000},
        },
    },
}

OUTPUT_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "overall_score",
        "criteria",
        "claims",
        "suggested_revision",
        "confidence",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise", "fail"]},
        "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "score", "notes"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "notes": {"type": "string"},
                },
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_text", "support", "evidence_ids", "notes"],
                "properties": {
                    "claim_text": {"type": "string"},
                    "support": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "unsupported",
                            "contradicted",
                            "uncertain",
                        ],
                    },
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        },
        "suggested_revision": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters}


STATUS_SCHEMA = _tool(
    "semantic_graph_status",
    "Report semantic-graph DB path, schema version, FTS status, and counts.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)

BEGIN_RUN_SCHEMA = _tool(
    "semantic_graph_begin_run",
    "Begin a semantic-graph analysis run with an objective and scope.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["objective"],
        "properties": {
            "objective": {"type": "string"},
            "title": {"type": "string"},
            "scope": {
                "type": "string",
                "enum": ["global", "project", "session", "run"],
                "default": "run",
            },
            "metadata": {"type": "object"},
        },
    },
)

INGEST_SCHEMA = _tool(
    "semantic_graph_ingest",
    "Ingest text as a sanitized provenance artifact; optionally extract a fragment.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "source_kind"],
        "properties": {
            "text": {"type": "string"},
            "source_kind": {
                "type": "string",
                "enum": [
                    "user_note",
                    "assistant_output",
                    "document",
                    "tool_result",
                    "meeting",
                    "research",
                    "code",
                    "other",
                ],
            },
            "title": {"type": "string"},
            "authority": {
                "type": "string",
                "enum": ["user", "assistant", "tool", "external", "system"],
            },
            "run_id": {"type": "string"},
            "extract": {"type": "boolean", "default": False},
            "subtype": {"type": "string"},
            "metadata": {"type": "object"},
        },
    },
)

SUBMIT_FRAGMENT_SCHEMA = _tool(
    "semantic_graph_submit_fragment",
    "Validate and store a typed graph fragment for a run.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["run_id", "producer_role", "fragment"],
        "properties": {
            "run_id": {"type": "string"},
            "producer_role": {"type": "string"},
            "fragment": {"type": "object"},
            "fragment_id": {"type": "string"},
            "model": {"type": "string"},
            "producer_id": {"type": "string"},
        },
    },
)

SEARCH_SCHEMA = _tool(
    "semantic_graph_search",
    "Search asserted/accepted graph nodes with optional filters.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "node_types": {
                "type": "array",
                "items": {"type": "string", "enum": _sorted(NODE_TYPES)},
            },
            "subtypes": {"type": "array", "items": {"type": "string"}},
            "statuses": {
                "type": "array",
                "items": {"type": "string", "enum": _sorted(STATUSES)},
            },
            "authorities": {
                "type": "array",
                "items": {"type": "string", "enum": _sorted(AUTHORITIES)},
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
            "include_evidence": {"type": "boolean"},
            "include_artifacts": {"type": "boolean"},
            "run_id": {"type": "string"},
        },
    },
)

GET_SCHEMA = _tool(
    "semantic_graph_get",
    "Fetch a run, node, edge, artifact, fragment, or evaluation by id.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["object_type", "object_id"],
        "properties": {
            "object_type": {
                "type": "string",
                "enum": ["run", "node", "edge", "artifact", "fragment", "evaluation"],
            },
            "object_id": {"type": "string"},
            "include_neighbors": {"type": "boolean"},
            "include_evidence": {"type": "boolean"},
            "max_neighbors": {"type": "integer", "minimum": 0, "maximum": 50},
        },
    },
)

FINALIZE_SCHEMA = _tool(
    "semantic_graph_finalize",
    "Finalize a run and optionally promote nodes under a strict policy.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["run_id"],
        "properties": {
            "run_id": {"type": "string"},
            "promotion_policy": {
                "type": "string",
                "enum": ["strict", "review_only"],
                "default": "strict",
            },
            "create_summary": {"type": "boolean", "default": True},
            "validate_only": {"type": "boolean", "default": False},
        },
    },
)

EVALUATE_OUTPUT_SCHEMA = _tool(
    "semantic_graph_evaluate_output",
    "Evaluate an artifact or text for groundedness; store evaluation without rewriting.",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "artifact_id": {"type": "string"},
            "text": {"type": "string"},
            "run_id": {"type": "string"},
            "criteria": {"type": "array", "items": {"type": "string"}},
            "reference_node_ids": {"type": "array", "items": {"type": "string"}},
            "store_result": {"type": "boolean", "default": True},
        },
    },
)

FEEDBACK_SCHEMA = _tool(
    "semantic_graph_feedback",
    "Apply user-confirmed accept/reject/supersede/correct feedback to a graph object.",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_type", "target_id", "action", "reason", "user_confirmed"],
        "properties": {
            "target_type": {
                "type": "string",
                "enum": ["node", "edge", "evaluation"],
            },
            "target_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["accept", "reject", "supersede", "correct"],
            },
            "reason": {"type": "string"},
            "replacement": {"type": "object"},
            "user_confirmed": {"type": "boolean"},
        },
    },
)

EXPORT_SCHEMA = _tool(
    "semantic_graph_export",
    "Export graph records to JSON, JSONL, or Markdown under the safe export root.",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "run_id": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["json", "jsonl", "markdown"],
                "default": "json",
            },
            "output_path": {"type": "string"},
            "include_artifacts": {"type": "boolean", "default": False},
            "include_rejected": {"type": "boolean", "default": False},
        },
    },
)

ALL_TOOL_SCHEMAS = [
    STATUS_SCHEMA,
    BEGIN_RUN_SCHEMA,
    INGEST_SCHEMA,
    SUBMIT_FRAGMENT_SCHEMA,
    SEARCH_SCHEMA,
    GET_SCHEMA,
    FINALIZE_SCHEMA,
    EVALUATE_OUTPUT_SCHEMA,
    FEEDBACK_SCHEMA,
    EXPORT_SCHEMA,
]

TOOL_NAMES = frozenset(s["name"] for s in ALL_TOOL_SCHEMAS)
