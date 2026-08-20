"""Experiential ledger for Ebbinghaus retrieval misses, events, and later AGIASI flows.

This module owns event / retrieval-attempt writes. It does not call the store's
public ``remember`` / ``recall`` APIs from inside transactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .models import (
    InsightStatus,
    InsightValidationResult,
    RevisionResult,
)
from .policies import EbbinghausPolicies

logger = logging.getLogger(__name__)

_MAX_EVENT_FIELD_CHARS = 4000
_MAX_EVENT_PAYLOAD_CHARS = 32000
_PREDICTION_ERROR_SOURCES = frozenset(
    {
        "user_correction",
        "explicit_contradiction",
        "tool_result",
        "expected_observed_mismatch",
        "high_surprise_latent_rescue",
        "new_time_qualified_evidence",
        "validation_failure",
    }
)
_STABILIZATION_EVIDENCE_TYPES = frozenset(
    {
        "user_confirmation",
        "unit_test",
        "integration_test",
        "trusted_tool_result",
        "external_source_validation",
        "correction_rehearsal_pass",
        "manual_validation",
    }
)


def normalize_query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _validated_sha256(value: str, *, name: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return digest


def _safe_json_payload(payload: Mapping[str, Any] | None) -> str:
    data = dict(payload or {})
    # Never persist opaque model scratch / chain-of-thought keys.
    blocked = {
        "chain_of_thought",
        "thinking",
        "reasoning",
        "scratchpad",
        "hidden_state",
        "raw_logits",
    }
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        key_text = str(key)
        if key_text.lower() in blocked:
            continue
        if isinstance(value, str) and len(value) > _MAX_EVENT_FIELD_CHARS:
            value = value[:_MAX_EVENT_FIELD_CHARS]
        cleaned[key_text] = value
    encoded = json.dumps(cleaned, ensure_ascii=False, default=str)
    if len(encoded) > _MAX_EVENT_PAYLOAD_CHARS:
        encoded = json.dumps(
            {"truncated": True, "keys": sorted(cleaned.keys())[:32]},
            ensure_ascii=False,
        )
    return encoded


class EbbinghausExperienceLedger:
    """Append-only experiential events and retrieval-attempt bookkeeping."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float],
        policies: EbbinghausPolicies,
        lock: threading.RLock,
    ) -> None:
        self._conn = conn
        self._now_fn = now_fn
        self.policies = policies
        self._lock = lock

    def record_event(
        self,
        event_type: str,
        *,
        memory_id: int | None = None,
        related_memory_id: int | None = None,
        belief_id: str = "",
        session_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_type),
                    memory_id,
                    related_memory_id,
                    str(belief_id or ""),
                    str(session_id or ""),
                    _safe_json_payload(payload),
                    float(self._now_fn()),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_prediction_error(
        self,
        *,
        memory_id: int,
        source: str,
        expected_hash: str,
        observed_hash: str,
        severity: float,
        requires_revision: bool,
        session_id: str = "",
    ) -> dict[str, Any]:
        source_key = str(source or "").strip().lower()
        if source_key not in _PREDICTION_ERROR_SOURCES:
            raise ValueError(
                "prediction error source must be one of "
                f"{sorted(_PREDICTION_ERROR_SOURCES)}"
            )
        expected = _validated_sha256(expected_hash, name="expected_hash")
        observed = _validated_sha256(observed_hash, name="observed_hash")
        if expected == observed:
            raise ValueError("expected_hash and observed_hash must differ")
        severity_value = float(severity)
        if not math.isfinite(severity_value) or not 0.0 <= severity_value <= 1.0:
            raise ValueError("severity must be between 0 and 1")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT memory_id, belief_id, belief_status, state, access_state "
                    "FROM memories WHERE memory_id = ?",
                    (int(memory_id),),
                ).fetchone()
                if row is None:
                    raise ValueError(f"memory_id {memory_id} not found")
                if str(row["belief_status"] or "current") != "current":
                    raise ValueError("prediction error requires the current belief version")
                if str(row["state"] or "active") != "active":
                    raise ValueError("prediction error requires an active belief")
                now = float(self._now_fn())
                self._conn.execute(
                    "UPDATE memories SET access_state = 'labile', updated_at = ? "
                    "WHERE memory_id = ?",
                    (now, int(memory_id)),
                )
                payload = {
                    "source": source_key,
                    "expected_hash": expected,
                    "observed_hash": observed,
                    "severity": severity_value,
                    "requires_revision": bool(requires_revision),
                }
                cur = self._conn.execute(
                    """
                    INSERT INTO memory_events(
                        event_type, memory_id, related_memory_id, belief_id,
                        session_id, payload, created_at
                    ) VALUES ('prediction_error', ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        int(memory_id),
                        str(row["belief_id"] or f"memory-{int(memory_id)}"),
                        str(session_id or ""),
                        _safe_json_payload(payload),
                        now,
                    ),
                )
                event_id = int(cur.lastrowid)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {
            "status": "labile",
            "event_id": event_id,
            "memory_id": int(memory_id),
            **payload,
        }

    def stabilize_memory(
        self,
        *,
        memory_id: int,
        evidence_type: str,
        evidence_hash: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        evidence_key = str(evidence_type or "").strip().lower()
        if evidence_key not in _STABILIZATION_EVIDENCE_TYPES:
            raise ValueError(
                "stabilization evidence type must be one of "
                f"{sorted(_STABILIZATION_EVIDENCE_TYPES)}"
            )
        digest = _validated_sha256(evidence_hash, name="evidence_hash")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT belief_id, belief_status, state, access_state "
                    "FROM memories WHERE memory_id = ?",
                    (int(memory_id),),
                ).fetchone()
                if row is None:
                    raise ValueError(f"memory_id {memory_id} not found")
                if (
                    str(row["belief_status"] or "current") != "current"
                    or str(row["state"] or "active") != "active"
                ):
                    raise ValueError("only an active current belief can be stabilized")
                if str(row["access_state"] or "accessible") != "labile":
                    raise ValueError("memory must be labile before stabilization")
                now = float(self._now_fn())
                self._conn.execute(
                    "UPDATE memories SET access_state = 'accessible', updated_at = ? "
                    "WHERE memory_id = ?",
                    (now, int(memory_id)),
                )
                payload = {
                    "evidence_type": evidence_key,
                    "evidence_hash": digest,
                }
                cur = self._conn.execute(
                    """
                    INSERT INTO memory_events(
                        event_type, memory_id, related_memory_id, belief_id,
                        session_id, payload, created_at
                    ) VALUES ('memory_stabilized', ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        int(memory_id),
                        str(row["belief_id"] or f"memory-{int(memory_id)}"),
                        str(session_id or ""),
                        _safe_json_payload(payload),
                        now,
                    ),
                )
                event_id = int(cur.lastrowid)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {
            "status": "stabilized",
            "event_id": event_id,
            "memory_id": int(memory_id),
            **payload,
        }

    def record_retrieval_miss(
        self,
        *,
        query_hash: str,
        query_excerpt: str,
        query_cues: Sequence[str],
        direct_best_score: float,
        session_id: str = "",
    ) -> int:
        with self._lock:
            self._trim_unresolved_misses_locked()
            cur = self._conn.execute(
                """
                INSERT INTO retrieval_attempts(
                    query_hash, query_excerpt, query_cues, outcome,
                    top_memory_id, matched_miss_id, result_memory_ids,
                    direct_best_score, rescue_score, surprise,
                    resolved_at, created_at
                ) VALUES (?, ?, ?, 'miss', NULL, NULL, '[]', ?, 0.0, 0.0, NULL, ?)
                """,
                (
                    str(query_hash),
                    str(query_excerpt or ""),
                    json.dumps(list(query_cues), ensure_ascii=False),
                    float(direct_best_score),
                    float(self._now_fn()),
                ),
            )
            attempt_id = int(cur.lastrowid)
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('retrieval_miss', NULL, NULL, '', ?, ?, ?)
                """,
                (
                    str(session_id or ""),
                    _safe_json_payload(
                        {
                            "attempt_id": attempt_id,
                            "query_hash": query_hash,
                            "direct_best_score": float(direct_best_score),
                            "cue_count": len(list(query_cues)),
                        }
                    ),
                    float(self._now_fn()),
                ),
            )
            self._conn.commit()
            return attempt_id

    def resolve_retrieval_miss(
        self,
        *,
        current_query_hash: str,
        current_cues: Sequence[str],
        rescued_memory_id: int,
        rescue_score: float,
        direct_best_score: float,
        session_id: str = "",
        current_attempt_id: int | None = None,
    ) -> dict[str, Any] | None:
        current_cue_set = {str(c).strip().lower() for c in current_cues if str(c).strip()}
        cutoff = float(self._now_fn()) - (
            float(self.policies.experience.miss_resolution_days) * 86400.0
        )
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT attempt_id, query_hash, query_cues, created_at
                FROM retrieval_attempts
                WHERE outcome = 'miss'
                  AND resolved_at IS NULL
                  AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (cutoff,),
            ).fetchall()
            best: tuple[int, float, float, int] | None = None
            for row in rows:
                attempt_id = int(row["attempt_id"] if isinstance(row, sqlite3.Row) else row[0])
                old_hash = str(row["query_hash"] if isinstance(row, sqlite3.Row) else row[1])
                raw_cues = row["query_cues"] if isinstance(row, sqlite3.Row) else row[2]
                created_at = float(row["created_at"] if isinstance(row, sqlite3.Row) else row[3])
                try:
                    old_cues = {str(c).strip().lower() for c in json.loads(raw_cues or "[]")}
                except (TypeError, ValueError, json.JSONDecodeError):
                    old_cues = set()
                same_hash = old_hash == str(current_query_hash)
                union = current_cue_set | old_cues
                overlap = (
                    len(current_cue_set & old_cues) / max(1, len(union))
                    if union
                    else 0.0
                )
                if not (same_hash or overlap >= 0.45):
                    continue
                prior_context = int(
                    current_attempt_id is not None
                    and attempt_id != int(current_attempt_id)
                )
                rank = (
                    prior_context,
                    overlap if not same_hash else 1.0,
                    created_at,
                    attempt_id,
                )
                if best is None or rank > best:
                    best = rank
            if best is None:
                return None
            attempt_id = best[3]
            resolution_gain = max(0.0, float(rescue_score) - float(direct_best_score))
            surprise = max(0.0, min(1.0, resolution_gain))
            now = float(self._now_fn())
            self._conn.execute(
                """
                UPDATE retrieval_attempts
                SET outcome = 'rescued',
                    matched_miss_id = attempt_id,
                    top_memory_id = ?,
                    result_memory_ids = ?,
                    rescue_score = ?,
                    surprise = ?,
                    resolved_at = ?
                WHERE attempt_id = ?
                """,
                (
                    int(rescued_memory_id),
                    json.dumps([int(rescued_memory_id)], ensure_ascii=False),
                    float(rescue_score),
                    float(surprise),
                    now,
                    attempt_id,
                ),
            )
            if (
                current_attempt_id is not None
                and int(current_attempt_id) != attempt_id
            ):
                self._conn.execute(
                    """
                    UPDATE retrieval_attempts
                    SET outcome = 'rescued',
                        matched_miss_id = ?,
                        top_memory_id = ?,
                        result_memory_ids = ?,
                        rescue_score = ?,
                        surprise = ?,
                        resolved_at = ?
                    WHERE attempt_id = ?
                    """,
                    (
                        attempt_id,
                        int(rescued_memory_id),
                        json.dumps([int(rescued_memory_id)], ensure_ascii=False),
                        float(rescue_score),
                        float(surprise),
                        now,
                        int(current_attempt_id),
                    ),
                )
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('retrieval_rescued', ?, NULL, '', ?, ?, ?)
                """,
                (
                    int(rescued_memory_id),
                    str(session_id or ""),
                    _safe_json_payload(
                        {
                            "attempt_id": attempt_id,
                            "rescue_score": float(rescue_score),
                            "surprise": float(surprise),
                        }
                    ),
                    now,
                ),
            )
            self._conn.commit()
            return {
                "attempt_id": attempt_id,
                "matched_miss_id": attempt_id,
                "surprise": surprise,
            }

    def revise_memory(
        self,
        *,
        memory_id: int,
        normalized_content: str,
        encoded: Mapping[str, Any],
        reason: str,
        evidence: Sequence[Mapping[str, Any]],
        confidence: float,
        test_query: str,
        source: str,
        session_id: str,
        tags: str = "",
        salience: float = 0.65,
        valence: float = 0.0,
        memory_type: str = "episodic",
    ) -> RevisionResult | dict[str, Any]:
        if int(memory_id) < 1:
            raise ValueError("memory_id must be >= 1")
        content = str(normalized_content or "").strip()
        if not content:
            raise ValueError("new_content must not be empty")
        reason_text = str(reason or "").strip()
        if not reason_text:
            raise ValueError("reason must not be empty")
        conf = float(confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                old = self._conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?",
                    (int(memory_id),),
                ).fetchone()
                if old is None:
                    raise ValueError(f"memory_id {memory_id} not found")

                # Resolve superseded pointer to current tip when needed.
                tip = old
                tip_id = int(old["memory_id"])
                if str(old["belief_status"] or "") == "superseded":
                    belief_id = str(old["belief_id"] or f"memory-{tip_id}")
                    current = self._conn.execute(
                        """
                        SELECT * FROM memories
                        WHERE belief_id = ? AND belief_status = 'current'
                        ORDER BY belief_version DESC, memory_id DESC
                        LIMIT 1
                        """,
                        (belief_id,),
                    ).fetchone()
                    if current is not None:
                        tip = current
                        tip_id = int(current["memory_id"])

                if str(tip["content"] or "").strip() == content:
                    self._conn.execute("COMMIT")
                    return {
                        "status": "idempotent",
                        "memory_id": tip_id,
                        "belief_id": str(tip["belief_id"] or f"memory-{tip_id}"),
                    }

                now = float(self._now_fn())
                belief_id = str(tip["belief_id"] or f"memory-{tip_id}")
                old_version = int(tip["belief_version"] or 1)
                new_version = old_version + 1
                cues = " ".join(list(encoded.get("cues") or [])[:32])

                self._conn.execute(
                    """
                    UPDATE memories
                    SET access_state = 'labile', updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, tip_id),
                )

                cur = self._conn.execute(
                    """
                    INSERT INTO memories (
                        content, encoded, cues, tags, salience, valence, strength,
                        source, session_id, created_at, updated_at,
                        last_rehearsed_at, state, last_anchor_at, memory_type,
                        belief_id, belief_version, belief_status, access_state,
                        confidence, supersedes_memory_id
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?,
                        ?, ?, 'current', 'accessible', ?, ?
                    )
                    """,
                    (
                        content,
                        json.dumps(dict(encoded), ensure_ascii=False),
                        cues,
                        tags or str(tip["tags"] or ""),
                        float(salience),
                        float(valence),
                        1.0 + float(salience),
                        source or "explicit_correction",
                        session_id or "",
                        now,
                        now,
                        now,
                        now,
                        memory_type or str(tip["memory_type"] or "episodic"),
                        belief_id,
                        new_version,
                        conf,
                        tip_id,
                    ),
                )
                new_id = int(cur.lastrowid)

                auto_archive = bool(self.policies.revision.auto_archive_superseded)
                self._conn.execute(
                    """
                    UPDATE memories
                    SET belief_status = 'superseded',
                        superseded_by_memory_id = ?,
                        access_state = 'latent',
                        state = CASE WHEN ? THEN 'archived' ELSE state END,
                        archived_at = CASE WHEN ? THEN ? ELSE archived_at END,
                        archive_reason = CASE WHEN ? THEN 'superseded' ELSE archive_reason END,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        new_id,
                        1 if auto_archive else 0,
                        1 if auto_archive else 0,
                        now,
                        1 if auto_archive else 0,
                        now,
                        tip_id,
                    ),
                )

                for relation_type in ("supersedes", "corrected_by"):
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_relations(
                            source_memory_id, target_memory_id, relation_type,
                            metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            new_id if relation_type == "supersedes" else tip_id,
                            tip_id if relation_type == "supersedes" else new_id,
                            relation_type,
                            _safe_json_payload({"reason": reason_text}),
                            now,
                        ),
                    )

                contested: list[int] = []
                if self.policies.revision.contest_dependents:
                    contested = self._contest_dependents_locked(
                        source_memory_id=tip_id,
                        reason=f"source_revised:{reason_text}",
                        visited=set(),
                    )

                query = str(test_query or "").strip()
                if not query:
                    query = " ".join(list(encoded.get("cues") or [])[:8])
                rehearsal_id = self._queue_correction_rehearsal_locked(
                    belief_id=belief_id,
                    old_memory_id=tip_id,
                    new_memory_id=new_id,
                    test_query=query,
                )

                self._conn.execute(
                    """
                    INSERT INTO memory_events(
                        event_type, memory_id, related_memory_id, belief_id,
                        session_id, payload, created_at
                    ) VALUES ('belief_revised', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id,
                        tip_id,
                        belief_id,
                        session_id or "",
                        _safe_json_payload(
                            {
                                "reason": reason_text,
                                "evidence": list(evidence or [])[:20],
                                "confidence": conf,
                                "old_version": old_version,
                                "new_version": new_version,
                            }
                        ),
                        now,
                    ),
                )
                self._conn.execute("COMMIT")
                return RevisionResult(
                    belief_id=belief_id,
                    old_memory_id=tip_id,
                    new_memory_id=new_id,
                    old_version=old_version,
                    new_version=new_version,
                    contested_memory_ids=contested,
                    queued_rehearsal_id=rehearsal_id,
                )
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def retract_memory(
        self,
        *,
        memory_id: int,
        reason: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        reason_text = str(reason or "").strip()
        if not reason_text:
            raise ValueError("reason must not be empty")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?",
                    (int(memory_id),),
                ).fetchone()
                if row is None:
                    raise ValueError(f"memory_id {memory_id} not found")
                now = float(self._now_fn())
                belief_id = str(row["belief_id"] or f"memory-{memory_id}")
                self._conn.execute(
                    """
                    UPDATE memories
                    SET belief_status = 'retracted',
                        state = 'archived',
                        access_state = 'latent',
                        archive_reason = 'retracted',
                        archived_at = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, now, int(memory_id)),
                )
                contested: list[int] = []
                if self.policies.revision.contest_dependents:
                    contested = self._contest_dependents_locked(
                        source_memory_id=int(memory_id),
                        reason=f"source_retracted:{reason_text}",
                        visited=set(),
                    )
                self._conn.execute(
                    """
                    INSERT INTO memory_events(
                        event_type, memory_id, related_memory_id, belief_id,
                        session_id, payload, created_at
                    ) VALUES ('belief_retracted', ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        int(memory_id),
                        belief_id,
                        session_id or "",
                        _safe_json_payload({"reason": reason_text}),
                        now,
                    ),
                )
                self._conn.execute("COMMIT")
                return {
                    "status": "retracted",
                    "memory_id": int(memory_id),
                    "belief_id": belief_id,
                    "contested_memory_ids": contested,
                }
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def belief_history(
        self,
        *,
        memory_id: int | None = None,
        belief_id: str = "",
    ) -> list[dict[str, Any]]:
        if memory_id is None and not belief_id:
            raise ValueError("memory_id or belief_id is required")
        with self._lock:
            resolved_belief = str(belief_id or "")
            if memory_id is not None:
                row = self._conn.execute(
                    "SELECT belief_id FROM memories WHERE memory_id = ?",
                    (int(memory_id),),
                ).fetchone()
                if row is None:
                    return []
                resolved_belief = str(row["belief_id"] or f"memory-{int(memory_id)}")
            rows = self._conn.execute(
                """
                SELECT memory_id, belief_id, belief_version, belief_status,
                       content, confidence, state, access_state,
                       supersedes_memory_id, superseded_by_memory_id,
                       created_at, updated_at
                FROM memories
                WHERE belief_id = ?
                ORDER BY belief_version ASC, memory_id ASC
                """,
                (resolved_belief,),
            ).fetchall()
            return [
                {
                    "memory_id": int(r["memory_id"]),
                    "belief_id": str(r["belief_id"] or ""),
                    "belief_version": int(r["belief_version"] or 1),
                    "belief_status": str(r["belief_status"] or "current"),
                    "content": str(r["content"] or ""),
                    "confidence": float(r["confidence"] or 0.0),
                    "state": str(r["state"] or "active"),
                    "access_state": str(r["access_state"] or "accessible"),
                    "supersedes_memory_id": r["supersedes_memory_id"],
                    "superseded_by_memory_id": r["superseded_by_memory_id"],
                    "created_at": float(r["created_at"] or 0.0),
                    "updated_at": float(r["updated_at"] or 0.0),
                }
                for r in rows
            ]

    def queue_correction_rehearsal(
        self,
        *,
        belief_id: str,
        old_memory_id: int,
        new_memory_id: int,
        test_query: str,
    ) -> int:
        with self._lock:
            return self._queue_correction_rehearsal_locked(
                belief_id=belief_id,
                old_memory_id=old_memory_id,
                new_memory_id=new_memory_id,
                test_query=test_query,
            )

    def run_correction_check(
        self,
        *,
        recall_fn: Callable[..., Any],
        rehearsal_id: int | None = None,
        limit: int = 1,
    ) -> dict[str, Any]:
        """Replay pending correction rehearsals without reinforcing retrieval."""
        with self._lock:
            if rehearsal_id is not None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM correction_rehearsals
                    WHERE rehearsal_id = ?
                    """,
                    (int(rehearsal_id),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM correction_rehearsals
                    WHERE status IN ('pending', 'failed')
                    ORDER BY created_at ASC, rehearsal_id ASC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()

            passed: list[int] = []
            failed: list[int] = []
            details: list[dict[str, Any]] = []
            now = float(self._now_fn())
            for row in rows:
                rid = int(row["rehearsal_id"])
                old_id = int(row["old_memory_id"])
                new_id = int(row["new_memory_id"])
                expected = int(row["expected_memory_id"] or new_id)
                query = str(row["test_query"] or "")
                outcome = recall_fn(
                    query,
                    limit=5,
                    reinforce=False,
                    include_archived=False,
                    include_history=False,
                    allow_rescue=False,
                    track=False,
                )
                result_ids = [int(item["memory_id"]) for item in outcome.results]
                top_id = result_ids[0] if result_ids else None
                old_recurred = old_id in result_ids
                ok = (
                    top_id == expected
                    and expected in result_ids
                    and not old_recurred
                )
                status = "passed" if ok else "failed"
                self._conn.execute(
                    """
                    UPDATE correction_rehearsals
                    SET status = ?,
                        old_error_recurred = ?,
                        actual_memory_ids = ?,
                        attempt_count = attempt_count + 1,
                        last_run_at = ?
                    WHERE rehearsal_id = ?
                    """,
                    (
                        status,
                        1 if old_recurred else 0,
                        json.dumps(result_ids, ensure_ascii=False),
                        now,
                        rid,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO memory_events(
                        event_type, memory_id, related_memory_id, belief_id,
                        session_id, payload, created_at
                    ) VALUES (?, ?, ?, ?, '', ?, ?)
                    """,
                    (
                        "correction_rehearsal_passed"
                        if ok
                        else "correction_rehearsal_failed",
                        new_id,
                        old_id,
                        str(row["belief_id"] or ""),
                        _safe_json_payload(
                            {
                                "rehearsal_id": rid,
                                "result_ids": result_ids,
                                "old_error_recurred": old_recurred,
                            }
                        ),
                        now,
                    ),
                )
                (passed if ok else failed).append(rid)
                details.append(
                    {
                        "rehearsal_id": rid,
                        "status": status,
                        "result_ids": result_ids,
                        "old_error_recurred": old_recurred,
                    }
                )
            self._conn.commit()
            return {
                "passed": passed,
                "failed": failed,
                "details": details,
            }

    def contest_dependents(
        self,
        *,
        source_memory_id: int,
        reason: str,
        visited: set[int] | None = None,
    ) -> list[int]:
        with self._lock:
            return self._contest_dependents_locked(
                source_memory_id=source_memory_id,
                reason=reason,
                visited=visited or set(),
            )

    def _queue_correction_rehearsal_locked(
        self,
        *,
        belief_id: str,
        old_memory_id: int,
        new_memory_id: int,
        test_query: str,
    ) -> int:
        now = float(self._now_fn())
        cur = self._conn.execute(
            """
            INSERT INTO correction_rehearsals(
                belief_id, old_memory_id, new_memory_id, test_query,
                expected_memory_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                str(belief_id),
                int(old_memory_id),
                int(new_memory_id),
                str(test_query or ""),
                int(new_memory_id),
                now,
            ),
        )
        return int(cur.lastrowid)

    def _contest_dependents_locked(
        self,
        *,
        source_memory_id: int,
        reason: str,
        visited: set[int],
    ) -> list[int]:
        if int(source_memory_id) in visited:
            return []
        visited.add(int(source_memory_id))
        contested: list[int] = []
        now = float(self._now_fn())

        semantic_rows = self._conn.execute(
            """
            SELECT DISTINCT semantic_memory_id AS memory_id
            FROM memory_provenance
            WHERE source_memory_id = ?
            """,
            (int(source_memory_id),),
        ).fetchall()
        for row in semantic_rows:
            mid = int(row["memory_id"])
            if mid in visited:
                continue
            self._conn.execute(
                """
                UPDATE memories
                SET belief_status = 'contested',
                    state = 'archived',
                    access_state = 'latent',
                    archive_reason = 'source_revised',
                    archived_at = COALESCE(archived_at, ?),
                    updated_at = ?
                WHERE memory_id = ?
                """,
                (now, now, mid),
            )
            contested.append(mid)
            contested.extend(
                self._contest_dependents_locked(
                    source_memory_id=mid, reason=reason, visited=visited
                )
            )

        insight_rows = self._conn.execute(
            """
            SELECT DISTINCT candidate_id, promoted_memory_id
            FROM insight_candidates
            WHERE candidate_id IN (
                SELECT candidate_id FROM insight_sources WHERE memory_id = ?
            )
            """,
            (int(source_memory_id),),
        ).fetchall()
        for row in insight_rows:
            candidate_id = str(row["candidate_id"])
            self._conn.execute(
                """
                UPDATE insight_candidates
                SET status = 'contested', updated_at = ?
                WHERE candidate_id = ?
                """,
                (now, candidate_id),
            )
            promoted = row["promoted_memory_id"]
            if promoted is not None:
                pmid = int(promoted)
                self._conn.execute(
                    """
                    UPDATE memories
                    SET belief_status = 'contested',
                        state = 'archived',
                        access_state = 'latent',
                        archive_reason = 'insight_contested',
                        archived_at = COALESCE(archived_at, ?),
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (now, now, pmid),
                )
                contested.append(pmid)
                contested.extend(
                    self._contest_dependents_locked(
                        source_memory_id=pmid, reason=reason, visited=visited
                    )
                )
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('insight_contested', ?, ?, '', '', ?, ?)
                """,
                (
                    int(promoted) if promoted is not None else None,
                    int(source_memory_id),
                    _safe_json_payload(
                        {"candidate_id": candidate_id, "reason": reason}
                    ),
                    now,
                ),
            )
        return contested

    def association_preview(self, *, limit: int) -> dict[str, Any]:
        """Build cue/tag bridge previews without promoting hypotheses to memory."""
        ip = self.policies.insight
        if not ip.enabled:
            return {"mode": "association_preview", "enabled": False, "associations": []}
        max_assoc = max(1, min(int(limit), int(ip.association_limit)))
        now = float(self._now_fn())
        associations: list[dict[str, Any]] = []

        focus_candidates: list[tuple[str, str, list[str]]] = []
        miss_rows = self._conn.execute(
            """
            SELECT attempt_id, query_cues FROM retrieval_attempts
            WHERE outcome = 'miss' AND resolved_at IS NULL
            ORDER BY created_at DESC LIMIT 8
            """
        ).fetchall()
        for row in miss_rows:
            try:
                cues = [str(c) for c in json.loads(row["query_cues"] or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError):
                cues = []
            focus_candidates.append(("retrieval_miss", str(row["attempt_id"]), cues))

        contested = self._conn.execute(
            """
            SELECT memory_id, cues FROM memories
            WHERE belief_status = 'contested'
            ORDER BY updated_at DESC LIMIT 4
            """
        ).fetchall()
        for row in contested:
            cues = str(row["cues"] or "").split()
            focus_candidates.append(("contested_belief", str(row["memory_id"]), cues))

        pool = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE belief_status IN ('current', 'context_dependent')
              AND access_state IN ('accessible', 'reactivated')
              AND state = 'active'
            ORDER BY salience DESC, updated_at DESC
            LIMIT 64
            """
        ).fetchall()

        for focus_type, focus_id, focus_cues in focus_candidates:
            if len(associations) >= max_assoc:
                break
            focus_tags: set[str] = set()
            focus_cue_set = {c.lower() for c in focus_cues if c}
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in pool:
                tags = {
                    t.lower()
                    for t in str(row["tags"] or "").replace(",", " ").split()
                    if t.strip()
                }
                cues = {c.lower() for c in str(row["cues"] or "").split() if c}
                tag_overlap = (
                    len(focus_tags & tags) / max(1, len(focus_tags | tags))
                    if focus_tags or tags
                    else 0.0
                )
                cue_overlap = (
                    len(focus_cue_set & cues) / max(1, len(focus_cue_set | cues))
                    if focus_cue_set or cues
                    else 0.0
                )
                if tag_overlap == 0.0 and cue_overlap == 0.0:
                    continue
                bridge = (
                    0.40 * tag_overlap
                    + 0.25 * cue_overlap
                    + 0.20 * (1.0 - cue_overlap)
                    + 0.15 * float(row["salience"] or 0.0)
                )
                scored.append((bridge, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            sources = []
            seen_content: set[str] = set()
            for _score, row in scored:
                content = str(row["content"] or "")
                if content in seen_content:
                    continue
                seen_content.add(content)
                sources.append(
                    {
                        "memory_id": int(row["memory_id"]),
                        "content": content,
                        "belief_status": str(row["belief_status"] or "current"),
                        "confidence": float(row["confidence"] or 0.0),
                    }
                )
                if len(sources) >= 6:
                    break
            if len(sources) < 2:
                continue
            association_id = f"assoc_{uuid.uuid4().hex}"
            source_ids = [int(s["memory_id"]) for s in sources]
            fingerprint = hashlib.sha256(
                ("|".join(map(str, sorted(source_ids))) + focus_type + focus_id).encode()
            ).hexdigest()
            expires_at = now + float(ip.candidate_ttl_days) * 86400.0
            payload = {
                "association_id": association_id,
                "focus": {"type": focus_type, "id": focus_id, "cues": focus_cues[:12]},
                "source_memories": sources,
                "instructions": [
                    "Propose one falsifiable hypothesis.",
                    "Do not present it as a fact.",
                    "List a validation method and possible counterexample.",
                ],
                "expires_at": expires_at,
            }
            try:
                self._conn.execute(
                    """
                    INSERT INTO association_previews(
                        association_id, focus_type, focus_id, source_memory_ids,
                        source_fingerprint, prompt_payload, status, expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'previewed', ?, ?)
                    """,
                    (
                        association_id,
                        focus_type,
                        focus_id,
                        json.dumps(source_ids, ensure_ascii=False),
                        fingerprint,
                        json.dumps(payload, ensure_ascii=False),
                        expires_at,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                continue
            associations.append(payload)

        self._conn.commit()
        return {
            "mode": "association_preview",
            "enabled": True,
            "associations": associations,
        }

    def propose_insight(
        self,
        *,
        association_id: str,
        hypothesis: str,
        source_memory_ids: Sequence[int],
        initial_confidence: float,
    ) -> dict[str, Any]:
        hyp = str(hypothesis or "").strip()
        if not hyp:
            raise ValueError("hypothesis must not be empty")
        conf = float(initial_confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("initial_confidence must be between 0 and 1")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM association_previews WHERE association_id = ?",
                (str(association_id),),
            ).fetchone()
            if row is None:
                raise ValueError("association_id not found")
            now = float(self._now_fn())
            if float(row["expires_at"] or 0.0) < now:
                raise ValueError("association preview expired")
            allowed = set(json.loads(row["source_memory_ids"] or "[]"))
            sources = [int(x) for x in source_memory_ids]
            if not sources or not set(sources).issubset(allowed):
                raise ValueError("source_memory_ids must be a non-empty subset of preview")
            candidate_id = f"insight_{uuid.uuid4().hex}"
            self._conn.execute(
                """
                INSERT INTO insight_candidates(
                    candidate_id, association_id, hypothesis, status,
                    initial_confidence, evidence, created_at, updated_at
                ) VALUES (?, ?, ?, 'candidate', ?, '[]', ?, ?)
                """,
                (candidate_id, str(association_id), hyp, conf, now, now),
            )
            for mid in sources:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO insight_sources(candidate_id, memory_id, source_role)
                    VALUES (?, ?, 'support')
                    """,
                    (candidate_id, int(mid)),
                )
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('insight_proposed', NULL, NULL, '', '', ?, ?)
                """,
                (
                    _safe_json_payload(
                        {
                            "candidate_id": candidate_id,
                            "association_id": association_id,
                            "hypothesis": hyp[:500],
                        }
                    ),
                    now,
                ),
            )
            self._conn.commit()
            return {
                "status": "candidate",
                "candidate_id": candidate_id,
                "association_id": association_id,
                "hypothesis": hyp,
                "initial_confidence": conf,
                "source_memory_ids": sources,
            }

    def validate_insight(
        self,
        *,
        candidate_id: str,
        validation_method: str,
        evidence: Sequence[Mapping[str, Any]],
        validated_confidence: float,
        summary: str,
        remember_fn: Callable[..., Mapping[str, Any]] | None = None,
    ) -> InsightValidationResult:
        from .models import ValidationMethod

        method = str(validation_method or "").strip()
        if method not in {m.value for m in ValidationMethod}:
            raise ValueError(f"unsupported validation_method: {method}")
        if not evidence:
            raise ValueError("evidence must contain at least one item")
        conf = float(validated_confidence)
        ip = self.policies.insight
        if conf < float(ip.validation_min_confidence):
            raise ValueError("validated_confidence below validation_min_confidence")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM insight_candidates WHERE candidate_id = ?",
                (str(candidate_id),),
            ).fetchone()
            if row is None:
                raise ValueError("candidate_id not found")
            if str(row["status"]) not in {"candidate", "validating"}:
                raise ValueError(f"cannot validate status={row['status']}")
            sources = self._conn.execute(
                "SELECT memory_id FROM insight_sources WHERE candidate_id = ?",
                (str(candidate_id),),
            ).fetchall()
            source_ids = [int(r["memory_id"]) for r in sources]
            for mid in source_ids:
                belief = self._conn.execute(
                    "SELECT belief_status FROM memories WHERE memory_id = ?",
                    (mid,),
                ).fetchone()
                if belief and str(belief["belief_status"]) in {
                    "contested",
                    "superseded",
                    "retracted",
                }:
                    raise ValueError("source memory is no longer current")
            now = float(self._now_fn())
            text = str(summary or row["hypothesis"] or "").strip()
            promoted_id = None
            if remember_fn is not None:
                remembered = remember_fn(
                    text,
                    tags=["insight", "validated", "semantic"],
                    salience=min(1.0, max(0.05, conf)),
                    source="validated_insight",
                    memory_type="semantic",
                )
                promoted_id = int(remembered["memory_id"])
                self._conn.execute(
                    "UPDATE memories SET confidence = ? WHERE memory_id = ?",
                    (conf, promoted_id),
                )
                for mid in source_ids:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_provenance(
                            semantic_memory_id, source_memory_id, relation, created_at
                        ) VALUES (?, ?, 'insight-derived', ?)
                        """,
                        (promoted_id, mid, now),
                    )
            self._conn.execute(
                """
                UPDATE insight_candidates
                SET status = 'validated',
                    validated_confidence = ?,
                    validation_method = ?,
                    evidence = ?,
                    promoted_memory_id = ?,
                    updated_at = ?
                WHERE candidate_id = ?
                """,
                (
                    conf,
                    method,
                    json.dumps(list(evidence), ensure_ascii=False, default=str),
                    promoted_id,
                    now,
                    str(candidate_id),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('insight_validated', ?, NULL, '', '', ?, ?)
                """,
                (
                    promoted_id,
                    _safe_json_payload(
                        {
                            "candidate_id": candidate_id,
                            "validation_method": method,
                            "validated_confidence": conf,
                        }
                    ),
                    now,
                ),
            )
            self._conn.commit()
            return InsightValidationResult(
                candidate_id=str(candidate_id),
                status=InsightStatus.VALIDATED,
                promoted_memory_id=promoted_id,
                contested_source_ids=[],
            )

    def reject_insight(self, *, candidate_id: str, reason: str) -> dict[str, Any]:
        reason_text = str(reason or "").strip()
        if not reason_text:
            raise ValueError("reason must not be empty")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM insight_candidates WHERE candidate_id = ?",
                (str(candidate_id),),
            ).fetchone()
            if row is None:
                raise ValueError("candidate_id not found")
            now = float(self._now_fn())
            self._conn.execute(
                """
                UPDATE insight_candidates
                SET status = 'rejected',
                    rejection_reason = ?,
                    updated_at = ?
                WHERE candidate_id = ?
                """,
                (reason_text, now, str(candidate_id)),
            )
            self._conn.execute(
                """
                INSERT INTO memory_events(
                    event_type, memory_id, related_memory_id, belief_id,
                    session_id, payload, created_at
                ) VALUES ('false_insight', NULL, NULL, '', '', ?, ?)
                """,
                (
                    _safe_json_payload(
                        {"candidate_id": candidate_id, "reason": reason_text}
                    ),
                    now,
                ),
            )
            self._conn.commit()
            return {
                "status": "rejected",
                "candidate_id": str(candidate_id),
                "reason": reason_text,
            }

    def _trim_unresolved_misses_locked(self) -> None:
        limit = int(self.policies.experience.max_unresolved_misses)
        count_row = self._conn.execute(
            """
            SELECT COUNT(*) AS count FROM retrieval_attempts
            WHERE outcome = 'miss' AND resolved_at IS NULL
            """
        ).fetchone()
        count = int(count_row["count"] if isinstance(count_row, sqlite3.Row) else count_row[0])
        overflow = count + 1 - limit
        if overflow <= 0:
            return
        old_ids = self._conn.execute(
            """
            SELECT attempt_id FROM retrieval_attempts
            WHERE outcome = 'miss' AND resolved_at IS NULL
            ORDER BY created_at ASC, attempt_id ASC
            LIMIT ?
            """,
            (overflow,),
        ).fetchall()
        for row in old_ids:
            attempt_id = int(row["attempt_id"] if isinstance(row, sqlite3.Row) else row[0])
            self._conn.execute(
                "DELETE FROM retrieval_attempts WHERE attempt_id = ?",
                (attempt_id,),
            )
