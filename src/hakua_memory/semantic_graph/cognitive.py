"""Pure cognitive rerank observations over an existing candidate list."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


_ACCESS_FACTORS = {
    "accessible": 1.00,
    "reactivated": 1.05,
    "labile": 0.90,
    "latent": 0.60,
}
_BELIEF_FACTORS = {
    "current": 1.00,
    "context_dependent": 0.95,
    "unverified": 0.75,
    "contested": 0.50,
    "superseded": 0.00,
    "retracted": 0.00,
}
_BELIEF_PRIORITY = {"current": 2, "context_dependent": 1}
_NONCURRENT = frozenset({"superseded", "retracted"})
_QUERY_MODES = frozenset({"normal", "history", "rescue"})


def _representative(
    links: Sequence[Mapping[str, Any]],
    states_by_memory: Mapping[int, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
    valid: list[Mapping[str, Any]] = []
    for link in links:
        try:
            memory_id = int(link["memory_id"])
            link_version = int(link["belief_version"])
            state = states_by_memory[memory_id]
            state_version = int(state["belief_version"])
            retention = float(state["projected_retention"])
            confidence = float(state["confidence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if state_version != link_version:
            continue
        if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
            continue
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            continue
        valid.append(state)
    if not valid:
        return None, []
    return max(
        valid,
        key=lambda state: (
            _BELIEF_PRIORITY.get(
                str(state.get("belief_status") or "").strip().lower(),
                0,
            ),
            float(state["projected_retention"]),
            float(state["confidence"]),
            int(state["belief_version"]),
            int(state["memory_id"]),
        ),
    ), valid


def observe_cognitive_rerank(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    links_by_node: Mapping[str, Sequence[Mapping[str, Any]]],
    states_by_memory: Mapping[int, Mapping[str, Any]],
    query_mode: str,
) -> list[dict[str, Any]]:
    """Add bounded shadow records without filtering or reordering candidates."""
    mode = str(query_mode or "normal").strip().lower()
    if mode not in _QUERY_MODES:
        raise ValueError("query_mode must be normal, history, or rescue")

    observed: list[dict[str, Any]] = []
    for base_rank, candidate in enumerate(ranked_candidates, start=1):
        node_id = str(candidate.get("node_id") or "")
        links = list(links_by_node.get(node_id) or [])
        representative, valid_states = _representative(links, states_by_memory)
        rank_score = 1.0 / (60.0 + base_rank)
        shadow: dict[str, Any] = {
            "base_rank": base_rank,
            "memory_link_count": len(links),
            "representative_memory_id": None,
            "projected_retention": None,
            "access_state": None,
            "belief_status": None,
            "cognitive_score": rank_score,
            "would_filter": False,
            "cognitive_rank": base_rank,
            "rank_changed": False,
            "reason": "unlinked" if not links else "stale_or_missing_state",
        }
        if representative is not None:
            memory_id = int(representative["memory_id"])
            retention = float(representative["projected_retention"])
            confidence = float(representative["confidence"])
            access_state = str(
                representative.get("access_state") or "accessible"
            ).strip().lower()
            belief_status = str(
                representative.get("belief_status") or "unverified"
            ).strip().lower()
            retention_factor = 0.75 + 0.25 * retention
            confidence_factor = 0.80 + 0.20 * confidence
            access_factor = _ACCESS_FACTORS.get(access_state, 1.00)
            belief_factor = _BELIEF_FACTORS.get(belief_status, 0.75)
            all_linked_latent = len(valid_states) == len(links) and all(
                str(state.get("access_state") or "").strip().lower() == "latent"
                for state in valid_states
            )
            noncurrent = belief_status in _NONCURRENT and mode != "history"
            latent_blocked = all_linked_latent and mode != "rescue"
            reason = "scored"
            if noncurrent:
                reason = "noncurrent_belief"
            elif latent_blocked:
                reason = "all_linked_latent"
            shadow.update(
                representative_memory_id=memory_id,
                projected_retention=retention,
                access_state=access_state,
                belief_status=belief_status,
                cognitive_score=(
                    rank_score
                    * retention_factor
                    * confidence_factor
                    * access_factor
                    * belief_factor
                ),
                would_filter=noncurrent or latent_blocked,
                reason=reason,
            )
        row = dict(candidate)
        row["cognitive_shadow"] = shadow
        observed.append(row)

    cognitive_order = sorted(
        range(len(observed)),
        key=lambda index: (
            -float(observed[index]["cognitive_shadow"]["cognitive_score"]),
            int(observed[index]["cognitive_shadow"]["base_rank"]),
            str(observed[index].get("node_id") or ""),
        ),
    )
    for cognitive_rank, index in enumerate(cognitive_order, start=1):
        shadow = observed[index]["cognitive_shadow"]
        shadow["cognitive_rank"] = cognitive_rank
        shadow["rank_changed"] = cognitive_rank != shadow["base_rank"]
    return observed


def activate_cognitive_rerank(
    observed_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the already-observed filter and rank without mutating input."""
    active = [
        dict(candidate)
        for candidate in observed_candidates
        if not candidate["cognitive_shadow"]["would_filter"]
    ]
    return sorted(
        active,
        key=lambda candidate: (
            int(candidate["cognitive_shadow"]["cognitive_rank"]),
            str(candidate.get("node_id") or ""),
        ),
    )


__all__ = ["activate_cognitive_rerank", "observe_cognitive_rerank"]
