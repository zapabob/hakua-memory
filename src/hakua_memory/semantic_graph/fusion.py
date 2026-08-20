"""Deterministic lexical/dense reciprocal-rank fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalCandidate:
    """Read-only candidate metadata produced by hybrid retrieval."""

    node_id: str
    lexical_rank: int | None
    dense_rank: int | None
    dense_similarity: float | None
    rrf_score: float
    source_count: int

    @property
    def best_rank(self) -> int:
        """Return the best one-based rank across available sources."""
        ranks = [rank for rank in (self.lexical_rank, self.dense_rank) if rank is not None]
        return min(ranks)


def _first_ranks(node_ids: Sequence[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, raw_node_id in enumerate(node_ids, start=1):
        node_id = str(raw_node_id)
        if node_id and node_id not in ranks:
            ranks[node_id] = rank
    return ranks


def reciprocal_rank_fusion(
    *,
    lexical_ids: Sequence[str],
    dense_ids: Sequence[str],
    k: int = 60,
    dense_similarities: dict[str, float] | None = None,
) -> list[RetrievalCandidate]:
    """Fuse one-based lexical and dense ranks without performing I/O.

    ``k`` is the RRF smoothing constant; result limiting belongs to the
    retrieval caller. Duplicate IDs retain their first rank in each source.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    lexical_ranks = _first_ranks(lexical_ids)
    dense_ranks = _first_ranks(dense_ids)
    all_ids = set(lexical_ranks) | set(dense_ranks)
    similarities = dense_similarities or {}
    candidates: list[RetrievalCandidate] = []
    for node_id in all_ids:
        lexical_rank = lexical_ranks.get(node_id)
        dense_rank = dense_ranks.get(node_id)
        ranks = [rank for rank in (lexical_rank, dense_rank) if rank is not None]
        score = math.fsum(1.0 / (k + rank) for rank in ranks)
        similarity = similarities.get(node_id) if dense_rank is not None else None
        if similarity is not None and not math.isfinite(float(similarity)):
            raise ValueError("dense similarity must be finite")
        candidates.append(
            RetrievalCandidate(
                node_id=node_id,
                lexical_rank=lexical_rank,
                dense_rank=dense_rank,
                dense_similarity=float(similarity) if similarity is not None else None,
                rrf_score=score,
                source_count=len(ranks),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate.rrf_score,
            -candidate.source_count,
            candidate.best_rank,
            candidate.node_id,
        )
    )
    return candidates


__all__ = ["RetrievalCandidate", "reciprocal_rank_fusion"]
