#!/usr/bin/env python3
"""Runnable sleep-cycle memory consolidation trial with tqdm progress.

Deterministic clock aging: remember -> decay -> sleep -> dream preview/apply.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hakua_memory.ebbinghaus.policies import (  # noqa: E402
    DreamPolicy,
    EbbinghausPolicies,
    ExperiencePolicy,
    SleepPolicy,
)
from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore  # noqa: E402


class _Clock:
    def __init__(self, start: float | None = None) -> None:
        self.now = float(start if start is not None else time.time())

    def __call__(self) -> float:
        return self.now

    def advance_days(self, days: float) -> None:
        self.now += float(days) * 86_400.0


def _age(store: EbbinghausMemoryStore, memory_id: int, days_ago: float) -> None:
    ts = store._now() - float(days_ago) * 86_400.0
    store._conn.execute(
        """
        UPDATE memories
        SET created_at = ?, updated_at = ?, last_rehearsed_at = ?,
            last_anchor_at = ?, last_sleep_at = NULL
        WHERE memory_id = ?
        """,
        (ts, ts, ts, ts, memory_id),
    )
    store._conn.commit()


def run_trial(*, experience: bool = False) -> dict:
    clock = _Clock()
    policies = EbbinghausPolicies(
        base_stability_days=3.0,
        sleep=SleepPolicy(
            rehearse_threshold=0.45,
            forget_threshold=0.12,
            salience_keep_threshold=0.70,
            limit=200,
            prune_mode="archive",
            max_sleep_rehearsals=0,
            recent_replay_limit=0,
            remote_integration_limit=0,
            max_negative_replay_per_budget=0,
        ),
        dreaming=DreamPolicy(enabled=True, min_source_count=2),
        experience=ExperiencePolicy(
            enabled=experience,
            correction_rehearsals_per_sleep=0,
        ),
    )
    with tempfile.TemporaryDirectory(prefix="hakua-sleep-trial-") as tmp:
        store = EbbinghausMemoryStore(
            Path(tmp) / "ebbinghaus.db",
            time_fn=clock,
            policies=policies,
        )
        try:
            episodes = [
                (
                    "Prefer short Japanese reports when summarizing incidents.",
                    ["user-pref", "communication"],
                    0.93,
                ),
                (
                    "Incident digests should stay concise and Japanese-first.",
                    ["user-pref", "communication"],
                    0.90,
                ),
                (
                    "Transient hallway chatter about the cafeteria menu.",
                    ["noise"],
                    0.18,
                ),
            ]
            ids: list[int] = []
            for content, tags, salience in tqdm(
                episodes, desc="remember", unit="mem"
            ):
                created = store.remember(content, tags=tags, salience=salience)
                ids.append(int(created["memory_id"]))

            for mid in tqdm(ids, desc="age_decay", unit="mem"):
                _age(store, mid, days_ago=50.0)

            sleep_report = store.sleep_cycle()
            preview = store.dream_preview()
            apply_report: dict | None = None
            if preview.get("clusters"):
                cluster = preview["clusters"][0]
                apply_report = store.dream_apply(
                    [
                        {
                            "cluster_id": cluster["cluster_id"],
                            "summary": (
                                "Consolidated lesson: keep Japanese incident "
                                "status digests short and operational."
                            ),
                            "source_memory_ids": cluster["source_memory_ids"],
                            "salience": 0.85,
                        }
                    ]
                )
            return {
                "sleep": {
                    "rehearsed": sleep_report.get("rehearsed"),
                    "forgotten": sleep_report.get("forgotten"),
                    "archived": sleep_report.get("archived"),
                    "latent": sleep_report.get("latent"),
                    "dream_candidates": sleep_report.get("dream_candidates"),
                },
                "dream_preview_clusters": len(preview.get("clusters") or []),
                "dream_apply": apply_report,
                "stats": store.stats(),
            }
        finally:
            store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_trial(experience=args.experience)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
