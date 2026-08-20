"""Embedding backend contracts for Semantic Graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot return a valid vector."""


@dataclass(frozen=True)
class EmbeddingModelIdentity:
    """Stable identity for one embedding representation namespace."""

    provider: str
    model: str
    revision: str
    dimensions: int
    serializer_version: int

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.serializer_version <= 0:
            raise ValueError("serializer_version must be positive")

    @property
    def namespace(self) -> str:
        revision = self.revision.strip() or "unversioned"
        return (
            f"{self.provider.strip()}:{self.model.strip()}:{revision}:"
            f"d{self.dimensions}:s{self.serializer_version}"
        )


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Backend capable of embedding queries and canonical node documents."""

    @property
    def identity(self) -> EmbeddingModelIdentity:
        ...

    def available(self) -> bool:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...
