"""Deterministic embedding backend used only by tests and benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .base import EmbeddingBackendError, EmbeddingModelIdentity


class DeterministicFakeEmbeddingBackend:
    """Return explicitly configured vectors without network or model calls."""

    def __init__(
        self,
        *,
        identity: EmbeddingModelIdentity,
        vectors: Mapping[str, Sequence[float]],
        is_available: bool = True,
        fail_on_embed: bool = False,
    ) -> None:
        self._identity = identity
        self._available = bool(is_available)
        self._fail_on_embed = bool(fail_on_embed)
        self._vectors: dict[str, tuple[float, ...]] = {}

        for text, vector in vectors.items():
            self._vectors[str(text)] = self._validate_vector(
                vector,
                label=f"vector for {text!r}",
            )

    @property
    def identity(self) -> EmbeddingModelIdentity:
        return self._identity

    def available(self) -> bool:
        return self._available

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        if not self._available:
            raise EmbeddingBackendError("fake embedding backend is unavailable")
        if self._fail_on_embed:
            raise EmbeddingBackendError("injected fake embedding failure")

        key = str(text)
        vector = self._vectors.get(key)
        if vector is None:
            raise EmbeddingBackendError(
                f"no deterministic vector configured for input: {key!r}"
            )
        return list(vector)

    def _validate_vector(
        self,
        vector: Sequence[float],
        *,
        label: str,
    ) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-numeric value") from exc

        if len(values) != self._identity.dimensions:
            raise ValueError(
                f"{label} has dimension {len(values)}; "
                f"expected {self._identity.dimensions}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label} contains NaN or infinity")
        return values
