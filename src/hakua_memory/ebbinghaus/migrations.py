"""Additive experiential-memory schema migrations for Ebbinghaus SQLite.

Migrations are idempotent, preserve existing memory rows, and use the SQLite
online backup API before applying version 1 DDL.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPERIENCE_SCHEMA_VERSION = 1
EXPERIENCE_MIGRATION_NAME = "experiential_memory_v1"

_EXPERIENCE_COLUMNS: list[tuple[str, str]] = [
    ("access_state", "TEXT NOT NULL DEFAULT 'accessible'"),
    ("belief_id", "TEXT NOT NULL DEFAULT ''"),
    ("belief_version", "INTEGER NOT NULL DEFAULT 1"),
    ("belief_status", "TEXT NOT NULL DEFAULT 'current'"),
    ("confidence", "REAL NOT NULL DEFAULT 0.60"),
    ("supersedes_memory_id", "INTEGER"),
    ("superseded_by_memory_id", "INTEGER"),
    ("latent_at", "REAL"),
    ("last_reactivated_at", "REAL"),
    ("reactivation_count", "INTEGER NOT NULL DEFAULT 0"),
]

_EXPERIENCE_DDL = """
CREATE TABLE IF NOT EXISTS ebbinghaus_schema_migrations (
    version       INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    applied_at    REAL NOT NULL,
    backup_path   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type         TEXT NOT NULL,
    memory_id          INTEGER,
    related_memory_id  INTEGER,
    belief_id          TEXT NOT NULL DEFAULT '',
    session_id         TEXT NOT NULL DEFAULT '',
    payload            TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_type_created
ON memory_events(event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_events_memory
ON memory_events(memory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_events_belief
ON memory_events(belief_id, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_attempts (
    attempt_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash          TEXT NOT NULL,
    query_excerpt       TEXT NOT NULL DEFAULT '',
    query_cues          TEXT NOT NULL DEFAULT '',
    outcome             TEXT NOT NULL,
    top_memory_id       INTEGER,
    matched_miss_id     INTEGER,
    result_memory_ids   TEXT NOT NULL DEFAULT '[]',
    direct_best_score   REAL NOT NULL DEFAULT 0.0,
    rescue_score        REAL NOT NULL DEFAULT 0.0,
    surprise            REAL NOT NULL DEFAULT 0.0,
    resolved_at         REAL,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_attempts_open_miss
ON retrieval_attempts(outcome, resolved_at, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_attempts_hash
ON retrieval_attempts(query_hash, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_relations (
    relation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id  INTEGER NOT NULL,
    target_memory_id  INTEGER NOT NULL,
    relation_type     TEXT NOT NULL,
    metadata          TEXT NOT NULL DEFAULT '{}',
    created_at        REAL NOT NULL,
    UNIQUE(source_memory_id, target_memory_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_relations_source
ON memory_relations(source_memory_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_memory_relations_target
ON memory_relations(target_memory_id, relation_type);

CREATE TABLE IF NOT EXISTS correction_rehearsals (
    rehearsal_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    belief_id          TEXT NOT NULL,
    old_memory_id      INTEGER NOT NULL,
    new_memory_id      INTEGER NOT NULL,
    test_query         TEXT NOT NULL,
    expected_memory_id INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'pending',
    old_error_recurred INTEGER NOT NULL DEFAULT 0,
    actual_memory_ids  TEXT NOT NULL DEFAULT '[]',
    attempt_count      INTEGER NOT NULL DEFAULT 0,
    last_run_at        REAL,
    created_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_correction_rehearsals_status
ON correction_rehearsals(status, created_at);

CREATE TABLE IF NOT EXISTS association_previews (
    association_id     TEXT PRIMARY KEY,
    focus_type         TEXT NOT NULL,
    focus_id           TEXT NOT NULL,
    source_memory_ids  TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL UNIQUE,
    prompt_payload     TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'previewed',
    expires_at         REAL NOT NULL,
    created_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS insight_candidates (
    candidate_id          TEXT PRIMARY KEY,
    association_id        TEXT NOT NULL,
    hypothesis            TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'candidate',
    initial_confidence    REAL NOT NULL,
    validated_confidence  REAL,
    validation_method     TEXT NOT NULL DEFAULT '',
    evidence              TEXT NOT NULL DEFAULT '[]',
    rejection_reason      TEXT NOT NULL DEFAULT '',
    promoted_memory_id    INTEGER,
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_insight_candidates_status
ON insight_candidates(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS insight_sources (
    candidate_id  TEXT NOT NULL,
    memory_id     INTEGER NOT NULL,
    source_role   TEXT NOT NULL DEFAULT 'support',
    PRIMARY KEY(candidate_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_insight_sources_memory
ON insight_sources(memory_id);

CREATE INDEX IF NOT EXISTS idx_memories_retrieval_state
ON memories(state, access_state, belief_status);

CREATE INDEX IF NOT EXISTS idx_memories_belief_version
ON memories(belief_id, belief_version DESC);

CREATE INDEX IF NOT EXISTS idx_memories_reactivation
ON memories(reactivation_count DESC, last_reactivated_at DESC);
"""


def verify_database_integrity(conn: sqlite3.Connection) -> None:
    """Raise if SQLite integrity or foreign-key checks fail."""
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or str(integrity[0]).lower() != "ok":
        raise sqlite3.DatabaseError(
            f"Ebbinghaus integrity_check failed: {integrity[0] if integrity else None}"
        )
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise sqlite3.DatabaseError(
            f"Ebbinghaus foreign_key_check failed to run: {exc}"
        ) from exc
    if fk_rows:
        raise sqlite3.DatabaseError(
            f"Ebbinghaus foreign_key_check failed: {len(fk_rows)} violation(s)"
        )


def create_online_backup(
    conn: sqlite3.Connection,
    db_path: Path,
    *,
    schema_version: int,
    now: float,
) -> Path | None:
    """Create an integrity-checked online backup beside ``db_path``."""
    db_path = Path(db_path)
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None

    stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(
        f"{db_path.stem}.pre-experience-v{schema_version}.{stamp}.bak"
    )
    if backup_path.exists():
        return backup_path

    destination = sqlite3.connect(str(backup_path))
    try:
        conn.backup(destination)
        row = destination.execute("PRAGMA integrity_check").fetchone()
        if row is None or str(row[0]).lower() != "ok":
            raise sqlite3.DatabaseError(
                "Ebbinghaus migration backup failed integrity_check"
            )
        destination.commit()
    except Exception:
        destination.close()
        backup_path.unlink(missing_ok=True)
        raise
    else:
        destination.close()
    return backup_path


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ebbinghaus_schema_migrations (
            version       INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            applied_at    REAL NOT NULL,
            backup_path   TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _memory_columns(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[1] if not isinstance(row, sqlite3.Row) else row["name"])
        for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }


def _add_experience_columns(conn: sqlite3.Connection) -> None:
    existing = _memory_columns(conn)
    for col_name, col_def in _EXPERIENCE_COLUMNS:
        if col_name in existing:
            continue
        conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
        logger.info("Ebbinghaus experience migration: added column %s", col_name)


def _backfill_experience_columns(conn: sqlite3.Connection) -> None:
    # Version-1 only: existing rows are pre-experience, so rewrite access_state
    # from storage state even when ADD COLUMN defaulted archived rows to accessible.
    conn.execute(
        """
        UPDATE memories
        SET access_state = CASE
            WHEN state = 'archived' THEN 'latent'
            ELSE 'accessible'
        END
        """
    )
    conn.execute(
        """
        UPDATE memories
        SET belief_id = 'memory-' || memory_id
        WHERE belief_id IS NULL OR belief_id = ''
        """
    )
    conn.execute(
        """
        UPDATE memories
        SET belief_version = 1
        WHERE belief_version IS NULL OR belief_version < 1
        """
    )
    conn.execute(
        """
        UPDATE memories
        SET belief_status = 'current'
        WHERE belief_status IS NULL OR belief_status = ''
        """
    )


def apply_experience_migrations(
    conn: sqlite3.Connection,
    db_path: Path,
    *,
    now: float,
) -> dict[str, Any]:
    """Apply additive experiential-memory schema migrations once."""
    db_path = Path(db_path)
    _ensure_migration_table(conn)
    conn.commit()

    existing = conn.execute(
        "SELECT version, backup_path FROM ebbinghaus_schema_migrations WHERE version = ?",
        (EXPERIENCE_SCHEMA_VERSION,),
    ).fetchone()
    if existing is not None:
        backup_path = ""
        if isinstance(existing, sqlite3.Row):
            backup_path = str(existing["backup_path"] or "")
        else:
            backup_path = str(existing[1] or "")
        return {
            "applied": False,
            "version": EXPERIENCE_SCHEMA_VERSION,
            "backup_path": backup_path,
        }

    backup = create_online_backup(
        conn,
        db_path,
        schema_version=EXPERIENCE_SCHEMA_VERSION,
        now=now,
    )
    backup_path = str(backup) if backup is not None else ""

    conn.execute("BEGIN IMMEDIATE")
    try:
        _add_experience_columns(conn)
        conn.executescript(_EXPERIENCE_DDL)
        _backfill_experience_columns(conn)
        conn.execute(
            """
            INSERT INTO ebbinghaus_schema_migrations(version, name, applied_at, backup_path)
            VALUES (?, ?, ?, ?)
            """,
            (
                EXPERIENCE_SCHEMA_VERSION,
                EXPERIENCE_MIGRATION_NAME,
                float(now),
                backup_path,
            ),
        )
        conn.execute("COMMIT")
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise sqlite3.DatabaseError(
            "Ebbinghaus experiential migration failed"
            + (f"; backup kept at {backup_path}" if backup_path else "")
        ) from exc

    verify_database_integrity(conn)
    logger.info(
        "Ebbinghaus experience migration v%s applied (backup=%s)",
        EXPERIENCE_SCHEMA_VERSION,
        backup_path or "none",
    )
    return {
        "applied": True,
        "version": EXPERIENCE_SCHEMA_VERSION,
        "backup_path": backup_path,
    }
