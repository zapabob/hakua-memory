"""SQLite store for the semantic-graph plugin (connection-per-operation)."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .embedding.base import EmbeddingModelIdentity
from .embedding.vectors import (
    FLOAT32_LE_DTYPE,
    cosine_similarity,
    l2_normalize,
    pack_float32_le,
    unpack_float32_le,
)
from .sanitize import sanitize_metadata, sanitize_value

logger = logging.getLogger("hakua_memory.semantic_graph")

DB_SCHEMA_VERSION = 3
GRAPH_SCHEMA_VERSION = 1
# Backwards-compatible alias retained for existing importers.
SCHEMA_VERSION = DB_SCHEMA_VERSION

DDL_CORE = """
CREATE TABLE IF NOT EXISTS graph_runs (
    run_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    objective TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT '',
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    finalized_at TEXT,
    summary_artifact_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    artifact_type TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    authority TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT '',
    truncated INTEGER NOT NULL DEFAULT 0,
    redaction_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session_turn ON artifacts(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(content_hash);
CREATE UNIQUE INDEX IF NOT EXISTS uq_artifact_dedupe
    ON artifacts(session_id, turn_id, artifact_type, content_hash);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    subtype TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL,
    normalized_label TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    identity_key TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    authority TEXT NOT NULL,
    confidence REAL NOT NULL,
    salience REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_type_status ON nodes(node_type, status);
CREATE INDEX IF NOT EXISTS idx_nodes_identity ON nodes(node_type, subtype, identity_key);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(normalized_label);

CREATE TABLE IF NOT EXISTS run_nodes (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, node_id),
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    relation_label TEXT NOT NULL DEFAULT '',
    strength REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY(target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_shape
    ON edges(source_node_id, target_node_id, edge_type, relation_label);

CREATE TABLE IF NOT EXISTS run_edges (
    run_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, edge_id),
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(edge_id) REFERENCES edges(edge_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_spans (
    evidence_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    node_id TEXT,
    edge_id TEXT,
    relation TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    quote TEXT NOT NULL,
    quote_hash TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY(edge_id) REFERENCES edges(edge_id) ON DELETE CASCADE,
    CHECK (
        (node_id IS NOT NULL AND edge_id IS NULL)
        OR
        (node_id IS NULL AND edge_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_evidence_artifact ON evidence_spans(artifact_id);
CREATE INDEX IF NOT EXISTS idx_evidence_node ON evidence_spans(node_id);
CREATE INDEX IF NOT EXISTS idx_evidence_edge ON evidence_spans(edge_id);

CREATE TABLE IF NOT EXISTS fragments (
    fragment_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    producer_role TEXT NOT NULL,
    producer_type TEXT NOT NULL,
    producer_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    created_at TEXT NOT NULL,
    UNIQUE(run_id, payload_hash),
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY,
    run_id TEXT,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    evaluator_role TEXT NOT NULL,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    criteria_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    suggested_revision TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_target ON evaluations(target_type, target_id);

CREATE TABLE IF NOT EXISTS graph_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT,
    event_type TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES graph_runs(run_id) ON DELETE SET NULL
);
"""

DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    node_id UNINDEXED,
    label,
    summary,
    tokenize='unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    artifact_id UNINDEXED,
    title,
    content,
    tokenize='unicode61'
);
"""

MIGRATION_V2_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS node_embeddings (
        node_id TEXT NOT NULL,
        namespace TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        revision TEXT NOT NULL DEFAULT '',
        serializer_version INTEGER NOT NULL CHECK(serializer_version > 0),
        dimensions INTEGER NOT NULL CHECK(dimensions > 0),
        dtype TEXT NOT NULL DEFAULT 'float32-le' CHECK(dtype = 'float32-le'),
        vector_blob BLOB NOT NULL CHECK(length(vector_blob) = dimensions * 4),
        source_text_hash TEXT NOT NULL CHECK(length(source_text_hash) = 64),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(node_id, namespace),
        FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_node_embeddings_namespace
        ON node_embeddings(namespace)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_node_embeddings_source_hash
        ON node_embeddings(namespace, source_text_hash)
    """,
)

MIGRATION_V3_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memory_node_links (
        memory_id INTEGER NOT NULL CHECK(memory_id > 0),
        node_id TEXT NOT NULL,
        belief_id TEXT NOT NULL DEFAULT '',
        belief_version INTEGER NOT NULL DEFAULT 1 CHECK(belief_version > 0),
        relation TEXT NOT NULL DEFAULT 'represents',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(memory_id, node_id, relation),
        FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_node_links_node
        ON memory_node_links(node_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_node_links_belief
        ON memory_node_links(belief_id, belief_version)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_state_cache (
        memory_id INTEGER PRIMARY KEY CHECK(memory_id > 0),
        belief_id TEXT NOT NULL DEFAULT '',
        belief_version INTEGER NOT NULL DEFAULT 1 CHECK(belief_version > 0),
        access_state TEXT NOT NULL,
        belief_status TEXT NOT NULL,
        memory_state TEXT NOT NULL,
        retention_at_sync REAL NOT NULL
            CHECK(retention_at_sync >= 0.0 AND retention_at_sync <= 1.0),
        stability_days REAL NOT NULL CHECK(stability_days > 0.0),
        salience REAL NOT NULL CHECK(salience >= 0.0 AND salience <= 1.0),
        valence REAL NOT NULL CHECK(valence >= -1.0 AND valence <= 1.0),
        confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
        protected INTEGER NOT NULL DEFAULT 0 CHECK(protected IN (0, 1)),
        source_updated_at REAL NOT NULL,
        synced_at REAL NOT NULL
    )
    """,
)

_NODE_EMBEDDING_COLUMNS = {
    "node_id", "namespace", "provider", "model", "revision",
    "serializer_version", "dimensions", "dtype", "vector_blob",
    "source_text_hash", "created_at", "updated_at",
}

_MEMORY_NODE_LINK_COLUMNS = {
    "memory_id", "node_id", "belief_id", "belief_version", "relation",
    "created_at", "updated_at",
}
_MEMORY_STATE_CACHE_COLUMNS = {
    "memory_id", "belief_id", "belief_version", "access_state",
    "belief_status", "memory_state", "retention_at_sync", "stability_days",
    "salience", "valence", "confidence", "protected", "source_updated_at",
    "synced_at",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dumps(obj: Any) -> str:
    return json.dumps(obj if obj is not None else {}, ensure_ascii=False, separators=(",", ":"))


def new_id() -> str:
    return uuid.uuid4().hex


class SemanticGraphStore:
    """Per-operation SQLite access. Never share a connection across threads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.fts_enabled = False
        self._ready = False
        self._lock = threading.Lock()
        self._local = threading.local()

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._migrate(conn)
            self._ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Share one connection with nested operations and rollback atomically."""
        self.ensure_ready()
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        with self._connect() as conn:
            self._local.connection = conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._local.connection = None

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version > DB_SCHEMA_VERSION:
            raise RuntimeError(
                "semantic-graph database schema "
                f"{version} is newer than supported schema {DB_SCHEMA_VERSION}"
            )

        if version < 1:
            self._create_schema_v1(conn)
            version = 1
        if version < 2:
            self._apply_migration(
                conn,
                target_version=2,
                statements=MIGRATION_V2_STATEMENTS,
            )
            version = 2
        if version < 3:
            self._apply_migration(
                conn,
                target_version=3,
                statements=MIGRATION_V3_STATEMENTS,
            )

        self._detect_fts(conn)

    def _create_schema_v1(self, conn: sqlite3.Connection) -> None:
        conn.executescript(DDL_CORE)
        try:
            conn.executescript(DDL_FTS)
        except sqlite3.Error as exc:
            logger.warning("semantic-graph: FTS5 unavailable, LIKE fallback: %s", exc)
        conn.execute("PRAGMA user_version = 1")

    def _apply_migration(
        self,
        conn: sqlite3.Connection,
        *,
        target_version: int,
        statements: tuple[str, ...],
    ) -> None:
        if conn.in_transaction:
            raise RuntimeError("schema migration must start outside an active transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                conn.execute(statement)
            self._validate_node_embeddings_schema(conn)
            if target_version >= 3:
                self._validate_cognitive_memory_schema(conn)
            conn.execute(f"PRAGMA user_version = {int(target_version)}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _validate_node_embeddings_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(node_embeddings)").fetchall()
        actual = {str(row["name"]) for row in rows}
        if actual != _NODE_EMBEDDING_COLUMNS:
            missing = sorted(_NODE_EMBEDDING_COLUMNS - actual)
            extra = sorted(actual - _NODE_EMBEDDING_COLUMNS)
            raise RuntimeError(
                "invalid node_embeddings schema: "
                f"missing={missing}, extra={extra}"
            )

        primary_key = sorted(
            (int(row["pk"]), str(row["name"]))
            for row in rows
            if int(row["pk"])
        )
        if primary_key != [(1, "node_id"), (2, "namespace")]:
            raise RuntimeError(
                "invalid node_embeddings schema: "
                "must have PRIMARY KEY(node_id, namespace)"
            )

        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='node_embeddings'"
        ).fetchone()
        table_sql = str(sql_row["sql"] or "").lower() if sql_row else ""
        required_constraints = (
            "check(serializer_version > 0)",
            "check(dimensions > 0)",
            "check(dtype = 'float32-le')",
            "check(length(vector_blob) = dimensions * 4)",
            "check(length(source_text_hash) = 64)",
        )
        if any(constraint not in table_sql for constraint in required_constraints):
            raise RuntimeError("invalid node_embeddings constraints")

        fk_rows = conn.execute("PRAGMA foreign_key_list(node_embeddings)").fetchall()
        has_node_fk = any(
            str(row["table"]) == "nodes"
            and str(row["from"]) == "node_id"
            and str(row["to"]) == "node_id"
            and str(row["on_delete"]).upper() == "CASCADE"
            for row in fk_rows
        )
        if not has_node_fk:
            raise RuntimeError(
                "node_embeddings must reference nodes(node_id) with ON DELETE CASCADE"
            )

    def _validate_cognitive_memory_schema(self, conn: sqlite3.Connection) -> None:
        for table, expected in (
            ("memory_node_links", _MEMORY_NODE_LINK_COLUMNS),
            ("memory_state_cache", _MEMORY_STATE_CACHE_COLUMNS),
        ):
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            actual = {str(row["name"]) for row in rows}
            if actual != expected:
                raise RuntimeError(
                    f"invalid {table} schema: "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )

        link_rows = conn.execute("PRAGMA table_info(memory_node_links)").fetchall()
        link_pk = sorted(
            (int(row["pk"]), str(row["name"]))
            for row in link_rows
            if int(row["pk"])
        )
        if link_pk != [
            (1, "memory_id"),
            (2, "node_id"),
            (3, "relation"),
        ]:
            raise RuntimeError(
                "memory_node_links must have "
                "PRIMARY KEY(memory_id, node_id, relation)"
            )

        fk_rows = conn.execute(
            "PRAGMA foreign_key_list(memory_node_links)"
        ).fetchall()
        has_node_fk = any(
            str(row["table"]) == "nodes"
            and str(row["from"]) == "node_id"
            and str(row["to"]) == "node_id"
            and str(row["on_delete"]).upper() == "CASCADE"
            for row in fk_rows
        )
        if not has_node_fk:
            raise RuntimeError(
                "memory_node_links must reference "
                "nodes(node_id) with ON DELETE CASCADE"
            )

    def _detect_fts(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone()
        self.fts_enabled = row is not None

    def get_status_counts(self) -> dict[str, Any]:
        self.ensure_ready()
        with self._connect() as conn:
            def _c(table: str) -> int:
                return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

            db_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            return {
                "schema_version": db_version,
                "graph_schema_version": GRAPH_SCHEMA_VERSION,
                "fts_enabled": self.fts_enabled,
                "runs": _c("graph_runs"),
                "nodes": _c("nodes"),
                "edges": _c("edges"),
                "artifacts": _c("artifacts"),
                "fragments": _c("fragments"),
                "evaluations": _c("evaluations"),
                "memory_node_links": _c("memory_node_links"),
                "memory_state_cache": _c("memory_state_cache"),
            }

    def create_run(
        self,
        *,
        objective: str,
        scope: str = "run",
        title: str = "",
        session_id: str = "",
        turn_id: str = "",
        model: str = "",
        platform: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        run_id = new_id()
        now = _utcnow()
        objective = sanitize_value(objective, max_chars=4000)
        scope = sanitize_value(scope, max_chars=128)
        title = sanitize_value(title, max_chars=500)
        session_id = sanitize_value(session_id, max_chars=256)
        turn_id = sanitize_value(turn_id, max_chars=256)
        model = sanitize_value(model, max_chars=256)
        platform = sanitize_value(platform, max_chars=256)
        metadata = sanitize_metadata(metadata or {})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO graph_runs("
                "run_id, title, objective, scope, status, session_id, turn_id, "
                "model, platform, schema_version, created_at, metadata_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    title or "",
                    objective,
                    scope,
                    "open",
                    session_id,
                    turn_id,
                    model,
                    platform,
                    GRAPH_SCHEMA_VERSION,
                    now,
                    _dumps(metadata or {}),
                ),
            )
        return {"run_id": run_id, "created_at": now, "status": "open"}

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM graph_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def upsert_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """Insert artifact; skip if same session/turn/type/hash already exists."""
        self.ensure_ready()
        artifact_id = artifact.get("artifact_id") or new_id()
        now = artifact.get("created_at") or _utcnow()
        artifact = dict(artifact)
        for key, limit in (("artifact_type", 128), ("title", 500), ("content", 12000), ("authority", 64), ("model", 256), ("platform", 256)):
            if key in artifact:
                artifact[key] = sanitize_value(artifact[key], max_chars=limit)
        artifact["metadata"] = sanitize_metadata(artifact.get("metadata") or {})
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT artifact_id FROM artifacts "
                "WHERE session_id=? AND turn_id=? AND artifact_type=? AND content_hash=?",
                (
                    artifact.get("session_id", ""),
                    artifact.get("turn_id", ""),
                    artifact.get("artifact_type", ""),
                    artifact.get("content_hash", ""),
                ),
            ).fetchone()
            if existing:
                return {
                    "artifact_id": existing["artifact_id"],
                    "duplicate": True,
                }
            conn.execute(
                "INSERT INTO artifacts("
                "artifact_id, run_id, artifact_type, title, content, content_hash, "
                "authority, session_id, turn_id, task_id, model, platform, "
                "truncated, redaction_count, metadata_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    artifact.get("run_id"),
                    artifact.get("artifact_type", "text"),
                    artifact.get("title", ""),
                    artifact.get("content", ""),
                    artifact.get("content_hash", ""),
                    artifact.get("authority", "assistant"),
                    artifact.get("session_id", ""),
                    artifact.get("turn_id", ""),
                    artifact.get("task_id", ""),
                    artifact.get("model", ""),
                    artifact.get("platform", ""),
                    1 if artifact.get("truncated") else 0,
                    int(artifact.get("redaction_count") or 0),
                    _dumps(artifact.get("metadata") or {}),
                    now,
                ),
            )
            if self.fts_enabled:
                conn.execute(
                    "INSERT INTO artifacts_fts(artifact_id, title, content) VALUES (?,?,?)",
                    (
                        artifact_id,
                        artifact.get("title", ""),
                        artifact.get("content", ""),
                    ),
                )
        return {"artifact_id": artifact_id, "duplicate": False}

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
            return dict(row) if row else None

    def upsert_node(self, node: dict[str, Any]) -> dict[str, Any]:
        self.ensure_ready()
        node = dict(node)
        for key, limit in (("node_type", 64), ("subtype", 128), ("label", 500), ("normalized_label", 500), ("summary", 4000), ("identity_key", 500), ("status", 32), ("authority", 64)):
            if key in node:
                node[key] = sanitize_value(node[key], max_chars=limit)
        node["metadata"] = sanitize_metadata(node.get("metadata") or {})
        node_id = node["node_id"]
        now = _utcnow()
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if existing:
                # Do not unconditionally raise confidence.
                new_conf = float(node.get("confidence", existing["confidence"]))
                old_conf = float(existing["confidence"])
                conf = min(new_conf, old_conf) if new_conf > old_conf else new_conf
                # Prefer non-empty higher-information summary without blind length win.
                summary = node.get("summary") or existing["summary"]
                if existing["summary"] and node.get("summary"):
                    if len(node["summary"]) <= len(existing["summary"]) * 1.1:
                        summary = existing["summary"] if existing["summary"] else node["summary"]
                meta_old = json.loads(existing["metadata_json"] or "{}")
                meta_new = node.get("metadata") or {}
                merged_meta = dict(meta_old)
                conflicts = dict(meta_old.get("metadata_conflicts") or {})
                for k, v in meta_new.items():
                    if k in merged_meta and merged_meta[k] != v:
                        conflicts[k] = {"old": merged_meta[k], "new": v}
                    else:
                        merged_meta[k] = v
                if conflicts:
                    merged_meta["metadata_conflicts"] = conflicts
                old_authority = str(existing["authority"])
                new_authority = str(node.get("authority", old_authority))
                authority_rank = {"assistant": 10, "subagent": 20, "tool": 30, "external": 40, "system": 50, "user": 60}
                if authority_rank.get(old_authority, 0) > authority_rank.get(new_authority, 0):
                    new_authority = old_authority
                old_status = str(existing["status"])
                new_status = str(node.get("status", old_status))
                if old_status in {"accepted", "asserted"} and new_status == "candidate":
                    new_status = old_status
                if old_status in {"rejected", "superseded"} and new_status not in {"rejected", "superseded"}:
                    new_status = old_status

                next_node_type = node.get("node_type", existing["node_type"])
                next_subtype = node.get("subtype", existing["subtype"])
                next_label = node.get("label", existing["label"])
                next_normalized_label = node.get(
                    "normalized_label", existing["normalized_label"]
                )
                next_identity_key = node.get("identity_key", existing["identity_key"])
                semantic_changed = (
                    str(existing["node_type"]),
                    str(existing["subtype"]),
                    str(existing["label"]),
                    str(existing["summary"]),
                    str(existing["identity_key"]),
                ) != (
                    str(next_node_type),
                    str(next_subtype),
                    str(next_label),
                    str(summary),
                    str(next_identity_key),
                )
                conn.execute(
                    "UPDATE nodes SET node_type=?, subtype=?, label=?, normalized_label=?, summary=?, "
                    "identity_key=?, status=?, authority=?, confidence=?, salience=?, "
                    "metadata_json=?, updated_at=? WHERE node_id=?",
                    (
                        next_node_type,
                        next_subtype,
                        next_label,
                        next_normalized_label,
                        summary,
                        next_identity_key,
                        new_status,
                        new_authority,
                        conf,
                        float(node.get("salience", existing["salience"])),
                        _dumps(merged_meta),
                        now,
                        node_id,
                    ),
                )
                if self.fts_enabled:
                    conn.execute("DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
                    conn.execute(
                        "INSERT INTO nodes_fts(node_id, label, summary) VALUES (?,?,?)",
                        (node_id, next_label, summary),
                    )
                if semantic_changed:
                    conn.execute(
                        "DELETE FROM node_embeddings WHERE node_id = ?",
                        (node_id,),
                    )
                return {"node_id": node_id, "updated": True}
            conn.execute(
                "INSERT INTO nodes("
                "node_id, node_type, subtype, label, normalized_label, summary, "
                "identity_key, status, authority, confidence, salience, "
                "metadata_json, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node_id,
                    node["node_type"],
                    node.get("subtype", ""),
                    node["label"],
                    node.get("normalized_label", ""),
                    node.get("summary", ""),
                    node.get("identity_key", ""),
                    node.get("status", "candidate"),
                    node.get("authority", "assistant"),
                    float(node.get("confidence", 0.5)),
                    float(node.get("salience", 0.5)),
                    _dumps(node.get("metadata") or {}),
                    now,
                    now,
                ),
            )
            if self.fts_enabled:
                conn.execute(
                    "INSERT INTO nodes_fts(node_id, label, summary) VALUES (?,?,?)",
                    (node_id, node["label"], node.get("summary", "")),
                )
        return {"node_id": node_id, "updated": False}

    def upsert_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        self.ensure_ready()
        edge = dict(edge)
        for key, limit in (("edge_type", 64), ("relation_label", 128), ("status", 32), ("rationale", 2000)):
            if key in edge:
                edge[key] = sanitize_value(edge[key], max_chars=limit)
        edge["metadata"] = sanitize_metadata(edge.get("metadata") or {})
        edge_id = edge["edge_id"]
        now = _utcnow()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT edge_id FROM edges WHERE source_node_id=? AND target_node_id=? "
                "AND edge_type=? AND relation_label=?",
                (
                    edge["source_node_id"],
                    edge["target_node_id"],
                    edge["edge_type"],
                    edge.get("relation_label", ""),
                ),
            ).fetchone()
            if existing:
                return {"edge_id": existing["edge_id"], "duplicate": True}
            conn.execute(
                "INSERT INTO edges("
                "edge_id, source_node_id, target_node_id, edge_type, relation_label, "
                "strength, confidence, status, rationale, metadata_json, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    edge_id,
                    edge["source_node_id"],
                    edge["target_node_id"],
                    edge["edge_type"],
                    edge.get("relation_label", ""),
                    float(edge.get("strength", 0.5)),
                    float(edge.get("confidence", 0.5)),
                    edge.get("status", "candidate"),
                    edge.get("rationale", ""),
                    _dumps(edge.get("metadata") or {}),
                    now,
                    now,
                ),
            )
        return {"edge_id": edge_id, "duplicate": False}

    def link_run_node(self, run_id: str, node_id: str, role: str = "member") -> None:
        self.ensure_ready()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO run_nodes(run_id, node_id, role, created_at) VALUES (?,?,?,?)",
                (run_id, node_id, role, _utcnow()),
            )

    def link_run_edge(self, run_id: str, edge_id: str) -> None:
        self.ensure_ready()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO run_edges(run_id, edge_id, created_at) VALUES (?,?,?)",
                (run_id, edge_id, _utcnow()),
            )

    def insert_evidence(self, evidence: dict[str, Any]) -> str:
        self.ensure_ready()
        evidence = dict(evidence)
        for key, limit in (("relation", 64), ("quote", 2000)):
            if key in evidence:
                evidence[key] = sanitize_value(evidence[key], max_chars=limit)
        eid = evidence.get("evidence_id") or new_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evidence_spans("
                "evidence_id, artifact_id, node_id, edge_id, relation, start_char, "
                "end_char, quote, quote_hash, confidence, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eid,
                    evidence["artifact_id"],
                    evidence.get("node_id"),
                    evidence.get("edge_id"),
                    evidence.get("relation", "supports"),
                    int(evidence["start_char"]),
                    int(evidence["end_char"]),
                    evidence.get("quote", ""),
                    evidence.get("quote_hash", ""),
                    float(evidence.get("confidence", 0.5)),
                    _utcnow(),
                ),
            )
        return eid

    def insert_fragment(self, fragment: dict[str, Any]) -> dict[str, Any]:
        self.ensure_ready()
        fragment = dict(fragment)
        for key, limit in (("producer_role", 128), ("producer_type", 128), ("producer_id", 256), ("model", 256), ("payload_json", 50000), ("status", 32)):
            if key in fragment:
                fragment[key] = sanitize_value(fragment[key], max_chars=limit)
        fid = fragment.get("fragment_id") or new_id()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO fragments("
                    "fragment_id, run_id, producer_role, producer_type, producer_id, "
                    "model, payload_json, payload_hash, status, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        fid,
                        fragment["run_id"],
                        fragment.get("producer_role", "unknown"),
                        fragment.get("producer_type", "subagent"),
                        fragment.get("producer_id", ""),
                        fragment.get("model", ""),
                        fragment.get("payload_json", "{}"),
                        fragment["payload_hash"],
                        fragment.get("status", "submitted"),
                        _utcnow(),
                    ),
                )
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT fragment_id FROM fragments WHERE run_id=? AND payload_hash=?",
                    (fragment["run_id"], fragment["payload_hash"]),
                ).fetchone()
                if row is None:
                    # Not a duplicate-payload hit (e.g. foreign-key violation).
                    raise
                return {
                    "success": True,
                    "duplicate": True,
                    "fragment_id": row["fragment_id"],
                }
        return {"success": True, "duplicate": False, "fragment_id": fid}

    def insert_evaluation(self, evaluation: dict[str, Any]) -> str:
        self.ensure_ready()
        evaluation = dict(evaluation)
        for key, limit in (("target_type", 64), ("evaluator_role", 128), ("verdict", 64), ("notes", 4000), ("suggested_revision", 8000)):
            if key in evaluation:
                evaluation[key] = sanitize_value(evaluation[key], max_chars=limit)
        evaluation["criteria"] = sanitize_value(evaluation.get("criteria") or {}, max_chars=4000)
        eid = evaluation.get("evaluation_id") or new_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluations("
                "evaluation_id, run_id, target_type, target_id, evaluator_role, "
                "verdict, score, criteria_json, notes, suggested_revision, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eid,
                    evaluation.get("run_id"),
                    evaluation["target_type"],
                    evaluation["target_id"],
                    evaluation.get("evaluator_role", "assistant"),
                    evaluation["verdict"],
                    float(evaluation.get("score", 0.0)),
                    _dumps(evaluation.get("criteria") or {}),
                    evaluation.get("notes", ""),
                    evaluation.get("suggested_revision", ""),
                    _utcnow(),
                ),
            )
        return eid

    def insert_event(self, event: dict[str, Any]) -> str:
        self.ensure_ready()
        event = dict(event)
        for key, limit in (("event_type", 128), ("actor_type", 64), ("actor_id", 256), ("session_id", 256), ("turn_id", 256), ("task_id", 256)):
            if key in event:
                event[key] = sanitize_value(event[key], max_chars=limit)
        event["payload"] = sanitize_value(event.get("payload") or {}, max_chars=4000)
        eid = event.get("event_id") or new_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO graph_events("
                "event_id, run_id, event_type, actor_type, actor_id, session_id, "
                "turn_id, task_id, payload_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    eid,
                    event.get("run_id"),
                    event["event_type"],
                    event.get("actor_type", "system"),
                    event.get("actor_id", ""),
                    event.get("session_id", ""),
                    event.get("turn_id", ""),
                    event.get("task_id", ""),
                    _dumps(event.get("payload") or {}),
                    _utcnow(),
                ),
            )
        return eid

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            return dict(row) if row else None

    def upsert_node_embedding(
        self,
        *,
        node_id: str,
        identity: EmbeddingModelIdentity,
        vector: Any,
        source_text_hash: str,
    ) -> dict[str, Any]:
        self.ensure_ready()
        if not re.fullmatch(r"[0-9a-f]{64}", source_text_hash):
            raise ValueError("source_text_hash must be a lowercase SHA-256 hex digest")
        blob = pack_float32_le(vector, expected_dimensions=identity.dimensions)
        now = _utcnow()
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM nodes WHERE node_id=?", (node_id,)).fetchone() is None:
                raise KeyError(f"unknown node_id: {node_id}")
            conn.execute(
                """
                INSERT INTO node_embeddings(
                    node_id, namespace, provider, model, revision,
                    serializer_version, dimensions, dtype, vector_blob,
                    source_text_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id, namespace) DO UPDATE SET
                    provider=excluded.provider, model=excluded.model,
                    revision=excluded.revision, serializer_version=excluded.serializer_version,
                    dimensions=excluded.dimensions, dtype=excluded.dtype,
                    vector_blob=excluded.vector_blob, source_text_hash=excluded.source_text_hash,
                    updated_at=excluded.updated_at
                """,
                (node_id, identity.namespace, identity.provider, identity.model, identity.revision,
                 identity.serializer_version, identity.dimensions, FLOAT32_LE_DTYPE, blob,
                 source_text_hash, now, now),
            )
        return {"node_id": node_id, "namespace": identity.namespace,
                "dimensions": identity.dimensions, "source_text_hash": source_text_hash}

    def get_node_embedding(
        self, *, node_id: str, namespace: str,
        expected_source_text_hash: str | None = None,
    ) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM node_embeddings WHERE node_id=? AND namespace=?",
                (node_id, namespace),
            ).fetchone()
        if row is None or (expected_source_text_hash is not None
                           and row["source_text_hash"] != expected_source_text_hash):
            return None
        result = dict(row)
        result["vector"] = unpack_float32_le(
            result.pop("vector_blob"), dimensions=int(result["dimensions"])
        )
        return result

    def search_node_embeddings_exact(
        self, *, namespace: str, query_vector: Any, top_k: int,
        min_similarity: float = -1.0, node_ids: Any = None,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        if top_k <= 0:
            return []
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        query = l2_normalize(query_vector)
        clauses = ["namespace = ?"]
        params: list[Any] = [namespace]
        if node_ids is not None:
            ids = sorted(set(node_ids))
            if not ids:
                return []
            clauses.append("node_id IN (" + ",".join("?" for _ in ids) + ")")
            params.extend(ids)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, namespace, dimensions, vector_blob, source_text_hash "
                "FROM node_embeddings WHERE " + " AND ".join(clauses) + " ORDER BY node_id",
                params,
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            dimensions = int(row["dimensions"])
            if dimensions != len(query):
                continue
            expected = expected_source_hashes.get(row["node_id"]) if expected_source_hashes else None
            if expected is not None and row["source_text_hash"] != expected:
                continue
            vector = unpack_float32_le(row["vector_blob"], dimensions=dimensions)
            score = cosine_similarity(query, vector)
            if score >= min_similarity:
                results.append({"node_id": row["node_id"], "namespace": row["namespace"],
                                "similarity": score, "source_text_hash": row["source_text_hash"]})
        results.sort(key=lambda item: (-float(item["similarity"]), str(item["node_id"])))
        return results[:top_k]

    def upsert_memory_node_link(self, link: dict[str, Any]) -> None:
        self.ensure_ready()
        memory_id = int(link["memory_id"])
        belief_version = int(link.get("belief_version", 1))
        if memory_id <= 0 or belief_version <= 0:
            raise ValueError("memory_id and belief_version must be positive")
        node_id = str(link["node_id"])
        belief_id = str(link.get("belief_id", ""))
        if not node_id or "\x00" in node_id or len(node_id) > 128:
            raise ValueError("memory node link node_id must be a valid opaque identifier")
        if not belief_id or "\x00" in belief_id or len(belief_id) > 256:
            raise ValueError("memory node link belief_id must be a valid opaque identifier")
        relation = sanitize_value(
            link.get("relation", "represents"),
            max_chars=64,
        )
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_node_links(
                    memory_id, node_id, belief_id, belief_version, relation,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, node_id, relation) DO UPDATE SET
                    belief_id = excluded.belief_id,
                    belief_version = excluded.belief_version,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    node_id,
                    belief_id,
                    belief_version,
                    relation,
                    now,
                    now,
                ),
            )

    def get_memory_node_links(
        self,
        *,
        memory_id: int | None = None,
        node_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        clauses: list[str] = []
        params: list[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(int(memory_id))
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(str(node_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_node_links"
                f"{where} ORDER BY memory_id, node_id, relation LIMIT ?",
                (*params, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_memory_state_cache(self, state: dict[str, Any]) -> None:
        self.ensure_ready()
        memory_id = int(state["memory_id"])
        belief_version = int(state.get("belief_version", 1))
        if memory_id <= 0 or belief_version <= 0:
            raise ValueError("memory_id and belief_version must be positive")
        numeric: dict[str, float] = {}
        for key in (
            "retention_at_sync",
            "stability_days",
            "salience",
            "valence",
            "confidence",
            "source_updated_at",
            "synced_at",
        ):
            value = float(state[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            numeric[key] = value
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_state_cache(
                    memory_id, belief_id, belief_version, access_state,
                    belief_status, memory_state, retention_at_sync,
                    stability_days, salience, valence, confidence, protected,
                    source_updated_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    belief_id = excluded.belief_id,
                    belief_version = excluded.belief_version,
                    access_state = excluded.access_state,
                    belief_status = excluded.belief_status,
                    memory_state = excluded.memory_state,
                    retention_at_sync = excluded.retention_at_sync,
                    stability_days = excluded.stability_days,
                    salience = excluded.salience,
                    valence = excluded.valence,
                    confidence = excluded.confidence,
                    protected = excluded.protected,
                    source_updated_at = excluded.source_updated_at,
                    synced_at = excluded.synced_at
                """,
                (
                    memory_id,
                    sanitize_value(state.get("belief_id", ""), max_chars=256),
                    belief_version,
                    sanitize_value(state["access_state"], max_chars=32),
                    sanitize_value(state["belief_status"], max_chars=32),
                    sanitize_value(state["memory_state"], max_chars=32),
                    numeric["retention_at_sync"],
                    numeric["stability_days"],
                    numeric["salience"],
                    numeric["valence"],
                    numeric["confidence"],
                    int(bool(state.get("protected", False))),
                    numeric["source_updated_at"],
                    numeric["synced_at"],
                ),
            )

    def get_memory_state_cache(self, memory_id: int) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_state_cache WHERE memory_id = ?",
                (int(memory_id),),
            ).fetchone()
        return dict(row) if row else None

    def list_memory_state_cache(self, *, limit: int = 5000) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_state_cache ORDER BY memory_id LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_edge(self, edge_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM edges WHERE edge_id=?", (edge_id,)).fetchone()
            return dict(row) if row else None

    def get_fragment(self, fragment_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM fragments WHERE fragment_id=?", (fragment_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_evaluation(self, evaluation_id: str) -> Optional[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_nodes(
        self,
        *,
        statuses: Optional[list[str]] = None,
        node_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        if node_types:
            clauses.append(f"node_type IN ({','.join('?' for _ in node_types)})")
            params.extend(node_types)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM nodes{where} ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_edges(self, *, include_rejected: bool = False, limit: int = 500) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            if include_rejected:
                rows = conn.execute(
                    "SELECT * FROM edges ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE status NOT IN ('rejected','superseded') "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def list_nodes_for_run(
        self,
        run_id: str,
        *,
        statuses: Optional[list[str]] = None,
        node_types: Optional[list[str]] = None,
        subtypes: Optional[list[str]] = None,
        authorities: Optional[list[str]] = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        clauses = ["rn.run_id = ?"]
        params: list[Any] = [run_id]
        for column, values in (("n.status", statuses), ("n.node_type", node_types), ("n.subtype", subtypes), ("n.authority", authorities)):
            if values:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                params.extend(values)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT n.* FROM nodes n JOIN run_nodes rn ON rn.node_id = n.node_id "
                f"WHERE {' AND '.join(clauses)} ORDER BY n.updated_at DESC LIMIT ?",
                (*params, max(1, limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_edges_for_run(
        self,
        run_id: str,
        *,
        include_rejected: bool = False,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        status_clause = "" if include_rejected else " AND e.status NOT IN ('rejected','superseded')"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.* FROM edges e JOIN run_edges re ON re.edge_id = e.edge_id "
                f"WHERE re.run_id = ?{status_clause} ORDER BY e.updated_at DESC LIMIT ?",
                (run_id, max(1, limit)),
            ).fetchall()
            return [dict(r) for r in rows]


    def list_evaluations_for_run(self, run_id: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at ASC", (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def list_artifacts(self, *, run_id: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def list_evidence_for_node(self, node_id: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_spans WHERE node_id=?", (node_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def neighbors(self, node_id: str, max_neighbors: int = 20) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE source_node_id=? OR target_node_id=? LIMIT ?",
                (node_id, node_id, max_neighbors),
            ).fetchall()
            return [dict(r) for r in rows]

    def search_nodes(
        self,
        query: str,
        *,
        statuses: Optional[list[str]] = None,
        node_types: Optional[list[str]] = None,
        subtypes: Optional[list[str]] = None,
        authorities: Optional[list[str]] = None,
        run_id: Optional[str] = None,
        top_k: int = 8,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        self.ensure_ready()
        statuses = statuses or ["asserted", "accepted"]
        q = (query or "").strip()
        if not q:
            return []
        terms = re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]", q.casefold())
        compact = "".join(terms)
        terms.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
        terms = list(dict.fromkeys(t for t in terms if len(t) >= 2))[:16] or [q]
        with self._connect() as conn:
            def extras(alias: str = "n") -> tuple[str, list[Any]]:
                clauses: list[str] = []
                params: list[Any] = []
                for column, values in ((f"{alias}.subtype", subtypes), (f"{alias}.authority", authorities)):
                    if values:
                        clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                        params.extend(values)
                if run_id:
                    clauses.append(f"{alias}.node_id IN (SELECT node_id FROM run_nodes WHERE run_id = ?)")
                    params.append(run_id)
                return (" AND " + " AND ".join(clauses)) if clauses else "", params

            if self.fts_enabled:
                try:
                    ascii_terms = [t for t in terms if t.isascii()]
                    fts_q = " OR ".join(f'"{t.replace(chr(34), " ")}"' for t in ascii_terms) or '"' + q.replace('"', ' ') + '"'
                    extra_sql, extra_params = extras()
                    rows = conn.execute(
                        "SELECT n.*, bm25(nodes_fts) AS bm25_score FROM nodes_fts "
                        "JOIN nodes n ON n.node_id = nodes_fts.node_id "
                        f"WHERE nodes_fts MATCH ? AND n.status IN ({','.join('?' for _ in statuses)}) "
                        "AND n.confidence >= ? "
                        + (f"AND n.node_type IN ({','.join('?' for _ in node_types)}) " if node_types else "")
                        + extra_sql + " ORDER BY bm25(nodes_fts) LIMIT ?",
                        (fts_q, *statuses, min_confidence, *(node_types or []), *extra_params, top_k * 3),
                    ).fetchall()
                    if rows:
                        return [dict(r) for r in rows]
                except sqlite3.Error:
                    pass
            term_sql = []
            term_params: list[Any] = []
            for term in terms:
                term_sql.append("(label LIKE ? OR summary LIKE ? OR identity_key LIKE ? OR subtype LIKE ?)")
                term_params.extend([f"%{term}%"] * 4)
            extra_sql, extra_params = extras(alias="nodes")
            rows = conn.execute(
                "SELECT *, 1.0 AS bm25_score FROM nodes WHERE (" + " OR ".join(term_sql) + ") "
                f"AND status IN ({','.join('?' for _ in statuses)}) AND confidence >= ? "
                + (f"AND node_type IN ({','.join('?' for _ in node_types)}) " if node_types else "")
                + extra_sql + " ORDER BY confidence DESC, salience DESC LIMIT ?",
                (*term_params, *statuses, min_confidence, *(node_types or []), *extra_params, top_k * 3),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_node_status(self, node_id: str, status: str) -> bool:
        self.ensure_ready()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE nodes SET status=?, updated_at=? WHERE node_id=?",
                (status, _utcnow(), node_id),
            )
            return cur.rowcount > 0

    def update_edge_status(self, edge_id: str, status: str) -> bool:
        self.ensure_ready()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE edges SET status=?, updated_at=? WHERE edge_id=?",
                (status, _utcnow(), edge_id),
            )
            return cur.rowcount > 0

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str = "finalized",
        summary_artifact_id: Optional[str] = None,
    ) -> None:
        self.ensure_ready()
        with self._connect() as conn:
            conn.execute(
                "UPDATE graph_runs SET status=?, finalized_at=?, summary_artifact_id=? "
                "WHERE run_id=?",
                (status, _utcnow(), summary_artifact_id, run_id),
            )

    def list_fragments_for_run(self, run_id: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fragments WHERE run_id=? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def purge_before(self, before_iso_date: str) -> dict[str, int]:
        """Delete rejected/superseded nodes and old artifacts before cutoff date."""
        self.ensure_ready()
        cutoff = before_iso_date.strip()
        if len(cutoff) == 10:
            cutoff = cutoff + "T00:00:00+00:00"
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                a = conn.execute(
                    "DELETE FROM artifacts WHERE created_at < ?", (cutoff,)
                ).rowcount
                n = conn.execute(
                    "DELETE FROM nodes WHERE status IN ('rejected','superseded') "
                    "AND updated_at < ?",
                    (cutoff,),
                ).rowcount
                e = conn.execute(
                    "DELETE FROM edges WHERE status IN ('rejected','superseded') "
                    "AND updated_at < ?",
                    (cutoff,),
                ).rowcount
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {"artifacts": a, "nodes": n, "edges": e}

    def vacuum(self) -> None:
        self.ensure_ready()
        with self._connect() as conn:
            conn.execute("VACUUM")
