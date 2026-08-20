"""Experiential-memory domain models for Ebbinghaus AGIASI extensions.

Wire values are stable contracts used by SQLite rows, tool payloads, and tests.
Do not rename enum members or change their ``.value`` strings without a
compatibility adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AccessState(str, Enum):
    ACCESSIBLE = "accessible"
    LATENT = "latent"
    REACTIVATED = "reactivated"
    LABILE = "labile"


class BeliefStatus(str, Enum):
    CURRENT = "current"
    CONTEXT_DEPENDENT = "context_dependent"
    UNVERIFIED = "unverified"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class RetrievalOutcome(str, Enum):
    HIT = "hit"
    MISS = "miss"
    RESCUED = "rescued"


class InsightStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    CONTESTED = "contested"
    EXPIRED = "expired"


class ValidationMethod(str, Enum):
    UNIT_TEST = "unit_test"
    EXTERNAL_SOURCE = "external_source"
    COUNTEREXAMPLE_SEARCH = "counterexample_search"
    USER_CONFIRMATION = "user_confirmation"
    INDEPENDENT_REASONING = "independent_reasoning"
    MANUAL = "manual"


@dataclass(frozen=True)
class RecallAttemptResult:
    query: str
    outcome: RetrievalOutcome
    results: list[dict[str, Any]]
    attempt_id: int | None = None
    matched_miss_id: int | None = None
    rescued_memory_id: int | None = None
    direct_best_score: float = 0.0
    rescue_score: float = 0.0
    surprise: float = 0.0
    state_note: str = ""


@dataclass(frozen=True)
class RevisionResult:
    belief_id: str
    old_memory_id: int
    new_memory_id: int
    old_version: int
    new_version: int
    contested_memory_ids: list[int] = field(default_factory=list)
    queued_rehearsal_id: int | None = None


@dataclass(frozen=True)
class InsightValidationResult:
    candidate_id: str
    status: InsightStatus
    promoted_memory_id: int | None
    contested_source_ids: list[int] = field(default_factory=list)
