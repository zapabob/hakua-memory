"""Sleep-cycle memory consolidation tests for hakua-memory.

Covers the biological-inspired pipeline:
remember -> age/decay -> sleep (rehearse / forget / dream-candidate) ->
dream preview/apply (semantic consolidation).
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from hakua_memory.composite import CompositeMemory
from hakua_memory.ebbinghaus.policies import (
    DreamPolicy,
    EbbinghausPolicies,
    ExperiencePolicy,
    SleepPolicy,
)
from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore


class _Clock:
    """Mutable monotonic clock for deterministic aging without real waits."""

    def __init__(self, start: float | None = None) -> None:
        self.now = float(start if start is not None else time.time())

    def __call__(self) -> float:
        return self.now

    def advance_days(self, days: float) -> None:
        self.now += float(days) * 86_400.0


def _policies(
    *,
    experience_enabled: bool = False,
    rehearse_threshold: float = 0.45,
    forget_threshold: float = 0.12,
    salience_keep_threshold: float = 0.70,
    max_sleep_rehearsals: int = 4,
) -> EbbinghausPolicies:
    return EbbinghausPolicies(
        base_stability_days=3.0,
        sleep=SleepPolicy(
            rehearse_threshold=rehearse_threshold,
            forget_threshold=forget_threshold,
            salience_keep_threshold=salience_keep_threshold,
            limit=200,
            prune_mode="archive",
            max_sleep_rehearsals=max_sleep_rehearsals,
            max_negative_sleep_rehearsals=1,
            recent_replay_limit=0,
            remote_integration_limit=0,
            max_negative_replay_per_budget=0,
        ),
        dreaming=DreamPolicy(enabled=True, min_source_count=2, max_clusters=4),
        experience=ExperiencePolicy(
            enabled=experience_enabled,
            functional_forgetting=True,
            latent_retention_threshold=0.12,
            archive_retention_threshold=0.03,
            latent_archive_after_days=30,
            correction_rehearsals_per_sleep=0,
        ),
    )


def _age_memory(store: EbbinghausMemoryStore, memory_id: int, *, days_ago: float) -> None:
    """Backdate anchors so forgetting_retention collapses without sleeping wall-clock."""
    ts = store._now() - float(days_ago) * 86_400.0
    store._conn.execute(
        """
        UPDATE memories
        SET created_at = ?,
            updated_at = ?,
            last_rehearsed_at = ?,
            last_anchor_at = ?,
            last_sleep_at = NULL
        WHERE memory_id = ?
        """,
        (ts, ts, ts, ts, memory_id),
    )
    store._conn.commit()


def _retention(store: EbbinghausMemoryStore, memory_id: int) -> float:
    row = store._conn.execute(
        "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    assert row is not None
    return store._retention(row)


def test_sleep_rehearses_high_salience_decayed_memory(tmp_path: Path) -> None:
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-rehearse.db",
        time_fn=clock,
        policies=_policies(max_sleep_rehearsals=4),
    )
    try:
        kept = store.remember(
            "Release freeze week starts 2026-09-20 for hakua-memory 0.3.5.",
            tags=["schedule", "release"],
            salience=0.95,
        )
        mid = int(kept["memory_id"])
        _age_memory(store, mid, days_ago=40)
        before = _retention(store, mid)
        assert before < 0.45

        report = store.sleep_cycle()
        assert mid in report["rehearsed"]
        assert mid not in report["forgotten"]
        after = store.get(mid)
        assert int(after["sleep_rehearsal_count"]) >= 1
        assert int(after["rehearsal_count"]) >= 1
        assert _retention(store, mid) > before
    finally:
        store.close()


def test_sleep_archives_low_salience_decayed_memory(tmp_path: Path) -> None:
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-forget.db",
        time_fn=clock,
        policies=_policies(),
    )
    try:
        noise = store.remember(
            "Transient hallway chatter about weather today.",
            tags=["noise", "ephemeral"],
            salience=0.20,
        )
        mid = int(noise["memory_id"])
        _age_memory(store, mid, days_ago=60)
        assert _retention(store, mid) < 0.12

        report = store.sleep_cycle(prune=True, prune_mode="archive")
        assert mid in report["forgotten"]
        assert mid in report["archived"]
        archived = store.get(mid)
        assert archived["state"] == "archived"
        reason = store._conn.execute(
            "SELECT archive_reason FROM memories WHERE memory_id = ?",
            (mid,),
        ).fetchone()
        assert reason is not None
        assert str(reason["archive_reason"]) == "sleep-forget"
    finally:
        store.close()


def test_sleep_marks_latent_when_experience_functional_forgetting(tmp_path: Path) -> None:
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-latent.db",
        time_fn=clock,
        policies=_policies(experience_enabled=True),
    )
    try:
        weak = store.remember(
            "Optional cafeteria menu note that is not operationally important.",
            tags=["noise"],
            salience=0.15,
        )
        mid = int(weak["memory_id"])
        _age_memory(store, mid, days_ago=60)
        report = store.sleep_cycle()
        assert mid in report["forgotten"]
        assert mid in report["latent"]
        row = store.get(mid)
        assert row["state"] == "active"
        assert row["access_state"] == "latent"
    finally:
        store.close()


def test_sleep_promotes_capped_rehearsals_to_dream_candidates(tmp_path: Path) -> None:
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-dream-cand.db",
        time_fn=clock,
        policies=_policies(max_sleep_rehearsals=0),
    )
    try:
        first = store.remember(
            "User prefers concise Japanese status updates for ops.",
            tags=["user-pref", "communication"],
            salience=0.92,
        )
        second = store.remember(
            "User wants Japanese bullet summaries in agent replies.",
            tags=["user-pref", "communication"],
            salience=0.91,
        )
        ids = [int(first["memory_id"]), int(second["memory_id"])]
        for mid in ids:
            _age_memory(store, mid, days_ago=45)
            assert _retention(store, mid) < 0.45

        report = store.sleep_cycle()
        for mid in ids:
            assert mid in report["dream_candidates"]
            row = store.get(mid)
            assert bool(row["dream_candidate"]) is True
    finally:
        store.close()


def test_dream_preview_and_apply_consolidates_sources(tmp_path: Path) -> None:
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "dream-apply.db",
        time_fn=clock,
        policies=_policies(max_sleep_rehearsals=0),
    )
    try:
        a = store.remember(
            "Prefer short Japanese reports when summarizing incidents.",
            tags=["user-pref", "communication"],
            salience=0.93,
        )
        b = store.remember(
            "Incident digests should stay concise and Japanese-first.",
            tags=["user-pref", "communication"],
            salience=0.90,
        )
        source_ids = [int(a["memory_id"]), int(b["memory_id"])]
        for mid in source_ids:
            _age_memory(store, mid, days_ago=50)

        sleep_report = store.sleep_cycle()
        assert set(source_ids).issubset(set(sleep_report["dream_candidates"]))

        preview = store.dream_preview()
        assert preview["enabled"] is True
        assert preview["clusters"], "expected at least one dream cluster"
        cluster = preview["clusters"][0]
        cluster_id = cluster["cluster_id"]
        assert len(cluster["source_memory_ids"]) >= 2

        applied = store.dream_apply(
            [
                {
                    "cluster_id": cluster_id,
                    "summary": "Consolidate: user prefers concise Japanese status digests.",
                    "tags": ["user-pref", "communication", "dream-summary"],
                    "salience": 0.85,
                    "source_memory_ids": cluster["source_memory_ids"],
                }
            ]
        )
        assert applied["enabled"] is True
        assert applied["applied"]
        semantic_id = int(applied["applied"][0]["semantic_memory_id"])
        semantic = store.get(semantic_id)
        assert semantic["memory_type"] == "semantic"
        tags = semantic.get("tags") or []
        assert "dream-summary" in tags

        for mid in cluster["source_memory_ids"]:
            source = store.get(int(mid))
            assert source["state"] == "archived"
            reason = store._conn.execute(
                "SELECT archive_reason FROM memories WHERE memory_id = ?",
                (int(mid),),
            ).fetchone()
            assert reason is not None
            assert str(reason["archive_reason"]) == "dream-consolidated"

        # Idempotent re-apply should not create a duplicate semantic lesson.
        again = store.dream_apply(
            [
                {
                    "cluster_id": cluster_id,
                    "summary": "Consolidate: user prefers concise Japanese status digests.",
                    "tags": ["user-pref", "communication", "dream-summary"],
                    "source_memory_ids": cluster["source_memory_ids"],
                }
            ]
        )
        again_id = int(
            again["applied"][0].get("semantic_memory_id")
            or again["applied"][0].get("memory_id")
        )
        assert again_id == semantic_id
    finally:
        store.close()


def test_composite_sleep_runs_full_consolidation_pipeline(tmp_path: Path) -> None:
    """Facade-level sleep + dream consolidation through CompositeMemory."""
    root = tmp_path / "composite-sleep"
    clock = _Clock()
    policies = _policies(max_sleep_rehearsals=0)
    # CompositeMemory constructs its own Ebbinghaus store; inject clock/policies.
    memory = CompositeMemory(root, enable_rag=False)
    memory.ebbinghaus.close()
    memory.ebbinghaus = EbbinghausMemoryStore(
        root / "ebbinghaus.db",
        time_fn=clock,
        policies=policies,
    )
    try:
        memory.remember(
            "Ops channel prefers Japanese one-liners after deploy.",
            tags=["user-pref", "ops"],
            salience=0.92,
        )
        memory.remember(
            "After deploy, keep Japanese one-line status for ops channel.",
            tags=["user-pref", "ops"],
            salience=0.90,
            valence=0.1,
        )
        rows = memory.ebbinghaus.list_memories(limit=10)
        assert len(rows) >= 2
        for row in rows:
            _age_memory(memory.ebbinghaus, int(row["memory_id"]), days_ago=55)

        sleep_report = memory.sleep()
        assert sleep_report["mode"] == "sleep_cycle"
        assert sleep_report["dream_candidates"]

        preview = memory.ebbinghaus.dream_preview()
        assert preview["clusters"]
        cluster = preview["clusters"][0]
        apply_report = memory.ebbinghaus.dream_apply(
            [
                {
                    "cluster_id": cluster["cluster_id"],
                    "summary": "Lesson: post-deploy ops updates stay Japanese and short.",
                    "source_memory_ids": cluster["source_memory_ids"],
                    "salience": 0.8,
                }
            ]
        )
        assert apply_report["applied"]
        stats = memory.ebbinghaus.stats()
        assert stats["dreaming_enabled"] is True
        assert int(stats.get("active_count") or 0) >= 1
    finally:
        memory.close()


def test_sleep_policies_are_immutable_replaceable() -> None:
    base = _policies()
    tweaked = replace(
        base,
        sleep=replace(base.sleep, rehearse_threshold=0.55),
    )
    assert tweaked.sleep.rehearse_threshold == 0.55
    assert base.sleep.rehearse_threshold == 0.45


def test_composite_remember_passes_salience_and_valence(tmp_path: Path) -> None:
    memory = CompositeMemory(tmp_path / "remember-passthrough", enable_rag=False)
    try:
        created = memory.remember(
            "High-value preference: keep deploy notes in Japanese.",
            tags=["user-pref"],
            salience=0.88,
            valence=0.25,
        )
        assert created["salience"] == pytest.approx(0.88)
        assert created["valence"] == pytest.approx(0.25)
        stored = memory.ebbinghaus.get(int(created["memory_id"]))
        assert stored["salience"] == pytest.approx(0.88)
        assert stored["valence"] == pytest.approx(0.25)
    finally:
        memory.close()


def _policies_with_replay(
    *,
    recent_replay_limit: int = 2,
    remote_integration_limit: int = 2,
) -> EbbinghausPolicies:
    return EbbinghausPolicies(
        base_stability_days=3.0,
        sleep=SleepPolicy(
            rehearse_threshold=0.45,
            forget_threshold=0.12,
            salience_keep_threshold=0.70,
            limit=200,
            prune_mode="archive",
            max_sleep_rehearsals=4,
            max_negative_sleep_rehearsals=1,
            recent_replay_limit=recent_replay_limit,
            remote_integration_limit=remote_integration_limit,
            max_negative_replay_per_budget=0,
        ),
        dreaming=DreamPolicy(enabled=True, min_source_count=2),
        experience=ExperiencePolicy(enabled=False, correction_rehearsals_per_sleep=0),
    )


def test_sleep_recent_replay_budget_on(tmp_path: Path) -> None:
    """Recent replay (age<=7d, high salience) fires when budget > 0."""
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-recent-replay.db",
        time_fn=clock,
        policies=_policies_with_replay(recent_replay_limit=2, remote_integration_limit=0),
    )
    try:
        recent = store.remember(
            "Fresh high-salience cue for recent replay budget.",
            tags=["ops", "hot"],
            salience=0.95,
        )
        mid = int(recent["memory_id"])
        # Age a few days so the cue is still "recent" (<=7d) but not brand-new noise.
        _age_memory(store, mid, days_ago=3.0)
        # Keep stability weak enough for recent-replay eligibility.
        store._conn.execute(
            "UPDATE memories SET strength = 0.6, rehearsal_count = 0 WHERE memory_id = ?",
            (mid,),
        )
        store._conn.commit()

        report = store.sleep_cycle()
        assert report["recent_replay"]["budget"] == 2
        assert mid in report["recent_replay"]["replayed"]
        assert mid in report["rehearsed"]
        assert int(store.get(mid)["sleep_rehearsal_count"]) >= 1
    finally:
        store.close()


def test_sleep_remote_integration_budget_on(tmp_path: Path) -> None:
    """Remote integration needs age>=30d, retrievals>=2, and provenance links."""
    clock = _Clock()
    store = EbbinghausMemoryStore(
        tmp_path / "sleep-remote-replay.db",
        time_fn=clock,
        policies=_policies_with_replay(recent_replay_limit=0, remote_integration_limit=2),
    )
    try:
        source = store.remember(
            "Long-lived source memory that already has retrieval history.",
            tags=["remote", "graph"],
            salience=0.85,
        )
        semantic = store.remember(
            "Linked semantic lesson derived from the remote source.",
            tags=["remote", "graph", "semantic"],
            salience=0.80,
            memory_type="semantic",
        )
        source_id = int(source["memory_id"])
        semantic_id = int(semantic["memory_id"])
        now = store._now()
        store._conn.execute(
            """
            INSERT INTO memory_provenance
                (semantic_memory_id, source_memory_id, relation, created_at)
            VALUES (?, ?, 'dream-derived', ?)
            """,
            (semantic_id, source_id, now),
        )
        store._conn.execute(
            """
            UPDATE memories
            SET retrieval_count = 3,
                created_at = ?,
                updated_at = ?,
                last_rehearsed_at = ?,
                last_anchor_at = ?,
                last_sleep_at = NULL
            WHERE memory_id = ?
            """,
            (
                now - 40 * 86_400.0,
                now - 40 * 86_400.0,
                now - 40 * 86_400.0,
                now - 40 * 86_400.0,
                source_id,
            ),
        )
        store._conn.commit()

        report = store.sleep_cycle()
        assert report["remote_integration"]["budget"] == 2
        assert source_id in report["remote_integration"]["replayed"]
        assert source_id in report["rehearsed"]
        assert int(store.get(source_id)["sleep_rehearsal_count"]) >= 1
    finally:
        store.close()
