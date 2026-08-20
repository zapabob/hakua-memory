"""Validated immutable policies for bounded Ebbinghaus memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class PolicyConfigError(ValueError):
    """Raised when plugin/config values are unsafe or inconsistent."""


def _clamp01(value: float, *, name: str) -> float:
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise PolicyConfigError(f"{name} must be in [0.0, 1.0], got {number}")
    return number


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    number = int(value)
    if number < minimum:
        raise PolicyConfigError(f"{name} must be >= {minimum}, got {number}")
    return number


def _as_tag_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [part.strip().lower() for part in raw.replace(";", ",").split(",")]
    elif isinstance(raw, Iterable):
        items = [str(item).strip().lower() for item in raw]
    else:
        items = [str(raw).strip().lower()]
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return tuple(cleaned)


@dataclass(frozen=True)
class SleepPolicy:
    rehearse_threshold: float = 0.35
    forget_threshold: float = 0.10
    salience_keep_threshold: float = 0.80
    limit: int = 1000
    prune_mode: str = "archive"
    max_sleep_rehearsals: int = 4
    max_negative_sleep_rehearsals: int = 1
    negative_valence_threshold: float = -0.60
    negative_reinforcement_multiplier: float = 0.25
    negative_prefetch_min_score: float = 0.28
    recent_replay_limit: int = 4
    remote_integration_limit: int = 4
    max_negative_replay_per_budget: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rehearse_threshold", _clamp01(self.rehearse_threshold, name="rehearse_threshold"))
        object.__setattr__(self, "forget_threshold", _clamp01(self.forget_threshold, name="forget_threshold"))
        object.__setattr__(
            self,
            "salience_keep_threshold",
            _clamp01(self.salience_keep_threshold, name="salience_keep_threshold"),
        )
        object.__setattr__(self, "limit", _positive_int(self.limit, name="limit"))
        mode = str(self.prune_mode or "archive").strip().lower()
        if mode not in {"none", "archive", "delete"}:
            raise PolicyConfigError(f"prune_mode must be none|archive|delete, got {mode!r}")
        object.__setattr__(self, "prune_mode", mode)
        object.__setattr__(
            self,
            "max_sleep_rehearsals",
            _positive_int(self.max_sleep_rehearsals, name="max_sleep_rehearsals", minimum=0),
        )
        object.__setattr__(
            self,
            "max_negative_sleep_rehearsals",
            _positive_int(
                self.max_negative_sleep_rehearsals,
                name="max_negative_sleep_rehearsals",
                minimum=0,
            ),
        )
        neg = float(self.negative_valence_threshold)
        if neg < -1.0 or neg > 0.0:
            raise PolicyConfigError(
                f"negative_valence_threshold must be in [-1.0, 0.0], got {neg}"
            )
        object.__setattr__(self, "negative_valence_threshold", neg)
        mult = float(self.negative_reinforcement_multiplier)
        if mult < 0.0 or mult > 1.0:
            raise PolicyConfigError(
                f"negative_reinforcement_multiplier must be in [0.0, 1.0], got {mult}"
            )
        object.__setattr__(self, "negative_reinforcement_multiplier", mult)
        object.__setattr__(
            self,
            "negative_prefetch_min_score",
            _clamp01(self.negative_prefetch_min_score, name="negative_prefetch_min_score"),
        )
        recent_limit = _positive_int(
            self.recent_replay_limit,
            name="recent_replay_limit",
            minimum=0,
        )
        if recent_limit > 32:
            raise PolicyConfigError(
                f"recent_replay_limit must be <= 32, got {recent_limit}"
            )
        object.__setattr__(self, "recent_replay_limit", recent_limit)
        remote_limit = _positive_int(
            self.remote_integration_limit,
            name="remote_integration_limit",
            minimum=0,
        )
        if remote_limit > 32:
            raise PolicyConfigError(
                f"remote_integration_limit must be <= 32, got {remote_limit}"
            )
        object.__setattr__(self, "remote_integration_limit", remote_limit)
        negative_limit = _positive_int(
            self.max_negative_replay_per_budget,
            name="max_negative_replay_per_budget",
            minimum=0,
        )
        if negative_limit > max(recent_limit, remote_limit):
            raise PolicyConfigError(
                "max_negative_replay_per_budget must not exceed either enabled "
                f"replay budget, got {negative_limit}"
            )
        object.__setattr__(
            self,
            "max_negative_replay_per_budget",
            negative_limit,
        )
        if self.forget_threshold >= self.rehearse_threshold:
            raise PolicyConfigError(
                "forget_threshold must be < rehearse_threshold "
                f"({self.forget_threshold} >= {self.rehearse_threshold})"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, *, defaults: "SleepPolicy | None" = None) -> "SleepPolicy":
        base = defaults or cls()
        data = dict(raw or {})
        return cls(
            rehearse_threshold=float(data.get("rehearse_threshold", base.rehearse_threshold)),
            forget_threshold=float(data.get("forget_threshold", base.forget_threshold)),
            salience_keep_threshold=float(
                data.get("salience_keep_threshold", base.salience_keep_threshold)
            ),
            limit=int(data.get("limit", base.limit)),
            prune_mode=str(data.get("prune_mode", base.prune_mode)),
            max_sleep_rehearsals=int(data.get("max_sleep_rehearsals", base.max_sleep_rehearsals)),
            max_negative_sleep_rehearsals=int(
                data.get("max_negative_sleep_rehearsals", base.max_negative_sleep_rehearsals)
            ),
            negative_valence_threshold=float(
                data.get("negative_valence_threshold", base.negative_valence_threshold)
            ),
            negative_reinforcement_multiplier=float(
                data.get(
                    "negative_reinforcement_multiplier",
                    base.negative_reinforcement_multiplier,
                )
            ),
            negative_prefetch_min_score=float(
                data.get("negative_prefetch_min_score", base.negative_prefetch_min_score)
            ),
            recent_replay_limit=int(
                data.get("recent_replay_limit", base.recent_replay_limit)
            ),
            remote_integration_limit=int(
                data.get(
                    "remote_integration_limit",
                    base.remote_integration_limit,
                )
            ),
            max_negative_replay_per_budget=int(
                data.get(
                    "max_negative_replay_per_budget",
                    base.max_negative_replay_per_budget,
                )
            ),
        )


@dataclass(frozen=True)
class CapacityPolicy:
    max_active_memories: int = 5000
    max_archived_memories: int = 20000
    archive_retention_days: int = 365
    protected_tags: tuple[str, ...] = (
        "pinned",
        "safety-critical",
        "user-profile",
        "consent",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_active_memories",
            _positive_int(self.max_active_memories, name="max_active_memories"),
        )
        object.__setattr__(
            self,
            "max_archived_memories",
            _positive_int(self.max_archived_memories, name="max_archived_memories"),
        )
        object.__setattr__(
            self,
            "archive_retention_days",
            _positive_int(self.archive_retention_days, name="archive_retention_days"),
        )
        tags = _as_tag_list(self.protected_tags)
        object.__setattr__(self, "protected_tags", tags)
        if self.max_active_memories >= self.max_active_memories + self.max_archived_memories:
            raise PolicyConfigError("max_active_memories capacity invariant failed")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "CapacityPolicy":
        data = dict(raw or {})
        return cls(
            max_active_memories=int(data.get("max_active_memories", 5000)),
            max_archived_memories=int(data.get("max_archived_memories", 20000)),
            archive_retention_days=int(data.get("archive_retention_days", 365)),
            protected_tags=tuple(
                data.get(
                    "protected_tags",
                    ("pinned", "safety-critical", "user-profile", "consent"),
                )
            ),
        )


@dataclass(frozen=True)
class DreamPolicy:
    enabled: bool = True
    candidate_limit: int = 24
    max_clusters: int = 8
    min_source_count: int = 2
    negative_summary_max_valence: float = -0.20
    archive_sources_after_apply: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "candidate_limit",
            _positive_int(self.candidate_limit, name="candidate_limit"),
        )
        object.__setattr__(
            self,
            "max_clusters",
            _positive_int(self.max_clusters, name="max_clusters"),
        )
        object.__setattr__(
            self,
            "min_source_count",
            _positive_int(self.min_source_count, name="min_source_count"),
        )
        valence = float(self.negative_summary_max_valence)
        if valence < -1.0 or valence > 0.0:
            raise PolicyConfigError(
                f"negative_summary_max_valence must be in [-1.0, 0.0], got {valence}"
            )
        object.__setattr__(self, "negative_summary_max_valence", valence)
        object.__setattr__(
            self,
            "archive_sources_after_apply",
            bool(self.archive_sources_after_apply),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DreamPolicy":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            candidate_limit=int(data.get("candidate_limit", 24)),
            max_clusters=int(data.get("max_clusters", 8)),
            min_source_count=int(data.get("min_source_count", 2)),
            negative_summary_max_valence=float(
                data.get("negative_summary_max_valence", -0.20)
            ),
            archive_sources_after_apply=bool(
                data.get("archive_sources_after_apply", True)
            ),
        )


@dataclass(frozen=True)
class ExperiencePolicy:
    enabled: bool = False
    functional_forgetting: bool = True
    latent_retention_threshold: float = 0.10
    archive_retention_threshold: float = 0.03
    latent_archive_after_days: int = 30
    rescue_enabled: bool = True
    rescue_min_score: float = 0.24
    miss_resolution_days: int = 90
    record_query_excerpt: bool = False
    record_hits: bool = False
    max_unresolved_misses: int = 2000
    resolved_attempt_retention_days: int = 90
    correction_rehearsals_per_sleep: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "functional_forgetting", bool(self.functional_forgetting))
        latent = _clamp01(self.latent_retention_threshold, name="latent_retention_threshold")
        archive = _clamp01(self.archive_retention_threshold, name="archive_retention_threshold")
        if not (0.0 <= archive < latent <= 1.0):
            raise PolicyConfigError(
                "archive_retention_threshold must be < latent_retention_threshold "
                f"({archive} >= {latent})"
            )
        object.__setattr__(self, "latent_retention_threshold", latent)
        object.__setattr__(self, "archive_retention_threshold", archive)
        object.__setattr__(
            self,
            "latent_archive_after_days",
            _positive_int(self.latent_archive_after_days, name="latent_archive_after_days"),
        )
        object.__setattr__(self, "rescue_enabled", bool(self.rescue_enabled))
        object.__setattr__(
            self,
            "rescue_min_score",
            _clamp01(self.rescue_min_score, name="rescue_min_score"),
        )
        object.__setattr__(
            self,
            "miss_resolution_days",
            _positive_int(self.miss_resolution_days, name="miss_resolution_days"),
        )
        object.__setattr__(self, "record_query_excerpt", bool(self.record_query_excerpt))
        object.__setattr__(self, "record_hits", bool(self.record_hits))
        object.__setattr__(
            self,
            "max_unresolved_misses",
            _positive_int(self.max_unresolved_misses, name="max_unresolved_misses"),
        )
        object.__setattr__(
            self,
            "resolved_attempt_retention_days",
            _positive_int(
                self.resolved_attempt_retention_days,
                name="resolved_attempt_retention_days",
            ),
        )
        object.__setattr__(
            self,
            "correction_rehearsals_per_sleep",
            _positive_int(
                self.correction_rehearsals_per_sleep,
                name="correction_rehearsals_per_sleep",
                minimum=0,
            ),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ExperiencePolicy":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            functional_forgetting=bool(data.get("functional_forgetting", True)),
            latent_retention_threshold=float(data.get("latent_retention_threshold", 0.10)),
            archive_retention_threshold=float(data.get("archive_retention_threshold", 0.03)),
            latent_archive_after_days=int(data.get("latent_archive_after_days", 30)),
            rescue_enabled=bool(data.get("rescue_enabled", True)),
            rescue_min_score=float(data.get("rescue_min_score", 0.24)),
            miss_resolution_days=int(data.get("miss_resolution_days", 90)),
            record_query_excerpt=bool(data.get("record_query_excerpt", False)),
            record_hits=bool(data.get("record_hits", False)),
            max_unresolved_misses=int(data.get("max_unresolved_misses", 2000)),
            resolved_attempt_retention_days=int(
                data.get("resolved_attempt_retention_days", 90)
            ),
            correction_rehearsals_per_sleep=int(
                data.get("correction_rehearsals_per_sleep", 10)
            ),
        )


@dataclass(frozen=True)
class RevisionPolicy:
    enabled: bool = True
    auto_archive_superseded: bool = True
    contest_dependents: bool = True
    protected_revision_allowed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "auto_archive_superseded", bool(self.auto_archive_superseded))
        object.__setattr__(self, "contest_dependents", bool(self.contest_dependents))
        object.__setattr__(
            self,
            "protected_revision_allowed",
            bool(self.protected_revision_allowed),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RevisionPolicy":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_archive_superseded=bool(data.get("auto_archive_superseded", True)),
            contest_dependents=bool(data.get("contest_dependents", True)),
            protected_revision_allowed=bool(data.get("protected_revision_allowed", True)),
        )


@dataclass(frozen=True)
class InsightPolicy:
    enabled: bool = True
    require_validation: bool = True
    association_limit: int = 8
    candidate_ttl_days: int = 30
    validation_min_confidence: float = 0.65

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "require_validation", bool(self.require_validation))
        limit = _positive_int(self.association_limit, name="association_limit")
        if limit > 64:
            raise PolicyConfigError(f"association_limit must be <= 64, got {limit}")
        object.__setattr__(self, "association_limit", limit)
        object.__setattr__(
            self,
            "candidate_ttl_days",
            _positive_int(self.candidate_ttl_days, name="candidate_ttl_days"),
        )
        object.__setattr__(
            self,
            "validation_min_confidence",
            _clamp01(self.validation_min_confidence, name="validation_min_confidence"),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "InsightPolicy":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            require_validation=bool(data.get("require_validation", True)),
            association_limit=int(data.get("association_limit", 8)),
            candidate_ttl_days=int(data.get("candidate_ttl_days", 30)),
            validation_min_confidence=float(data.get("validation_min_confidence", 0.65)),
        )


@dataclass(frozen=True)
class EbbinghausPolicies:
    base_stability_days: float = 3.0
    decay_threshold: float = 0.10
    max_prefetch: int = 5
    min_prefetch_score: float = 0.18
    auto_encode_turns: bool = False
    sleep: SleepPolicy = field(default_factory=SleepPolicy)
    capacity: CapacityPolicy = field(default_factory=CapacityPolicy)
    dreaming: DreamPolicy = field(default_factory=DreamPolicy)
    experience: ExperiencePolicy = field(default_factory=ExperiencePolicy)
    revision: RevisionPolicy = field(default_factory=RevisionPolicy)
    insight: InsightPolicy = field(default_factory=InsightPolicy)

    def __post_init__(self) -> None:
        base = float(self.base_stability_days)
        if base <= 0:
            raise PolicyConfigError("base_stability_days must be > 0")
        object.__setattr__(self, "base_stability_days", base)
        object.__setattr__(self, "decay_threshold", _clamp01(self.decay_threshold, name="decay_threshold"))
        object.__setattr__(self, "max_prefetch", _positive_int(self.max_prefetch, name="max_prefetch"))
        object.__setattr__(
            self,
            "min_prefetch_score",
            _clamp01(self.min_prefetch_score, name="min_prefetch_score"),
        )
        object.__setattr__(self, "auto_encode_turns", bool(self.auto_encode_turns))

    @classmethod
    def from_plugin_config(cls, config: dict[str, Any] | None) -> "EbbinghausPolicies":
        data = dict(config or {})
        sleep_raw = data.get("sleep") if isinstance(data.get("sleep"), dict) else {}
        capacity_raw = data.get("capacity") if isinstance(data.get("capacity"), dict) else {}
        dream_raw = data.get("dreaming") if isinstance(data.get("dreaming"), dict) else {}
        experience_raw = data.get("experience") if isinstance(data.get("experience"), dict) else {}
        revision_raw = data.get("revision") if isinstance(data.get("revision"), dict) else {}
        insight_raw = data.get("insight") if isinstance(data.get("insight"), dict) else {}
        return cls(
            base_stability_days=float(data.get("base_stability_days", 3.0)),
            decay_threshold=float(data.get("decay_threshold", 0.10)),
            max_prefetch=int(data.get("max_prefetch", 5)),
            min_prefetch_score=float(data.get("min_prefetch_score", 0.18)),
            auto_encode_turns=bool(data.get("auto_encode_turns", False)),
            sleep=SleepPolicy.from_mapping(sleep_raw),
            capacity=CapacityPolicy.from_mapping(capacity_raw),
            dreaming=DreamPolicy.from_mapping(dream_raw),
            experience=ExperiencePolicy.from_mapping(experience_raw),
            revision=RevisionPolicy.from_mapping(revision_raw),
            insight=InsightPolicy.from_mapping(insight_raw),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "EbbinghausPolicies":
        """Spec-facing alias for :meth:`from_plugin_config`."""
        return cls.from_plugin_config(config)


def resolve_prune_mode(*, prune: bool | None, prune_mode: str | None, default: str) -> str:
    """Resolve prune_mode with legacy prune=True meaning physical delete."""
    if prune_mode is not None and str(prune_mode).strip():
        mode = str(prune_mode).strip().lower()
        if mode not in {"none", "archive", "delete"}:
            raise PolicyConfigError(f"prune_mode must be none|archive|delete, got {mode!r}")
        return mode
    if prune is True:
        return "delete"
    if prune is False:
        return "none"
    return str(default or "archive").lower()


def is_protected(tags: Sequence[str], protected_tags: Sequence[str]) -> bool:
    tag_set = {str(tag).strip().lower() for tag in tags}
    return bool(tag_set & {str(tag).strip().lower() for tag in protected_tags})
