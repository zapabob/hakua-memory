"""Validated float32-le vector operations for Semantic Graph embeddings."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from .base import EmbeddingBackendError


FLOAT32_LE_DTYPE = "float32-le"
_FLOAT32_SIZE = 4


class EmbeddingVectorError(EmbeddingBackendError):
    """Raised when an embedding vector is malformed or unusable."""


def validate_vector(
    values: Sequence[float],
    *,
    expected_dimensions: int | None = None,
    require_nonzero: bool = True,
) -> tuple[float, ...]:
    """Convert values to finite floats and enforce dimensional invariants."""
    if expected_dimensions is not None and expected_dimensions <= 0:
        raise ValueError("expected_dimensions must be positive")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise EmbeddingVectorError("embedding contains a non-numeric value") from exc
    if not vector:
        raise EmbeddingVectorError("embedding vector must not be empty")
    if expected_dimensions is not None and len(vector) != expected_dimensions:
        raise EmbeddingVectorError(
            f"embedding dimension {len(vector)} does not match expected {expected_dimensions}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingVectorError("embedding contains NaN or infinity")
    scale = max(abs(value) for value in vector)
    if require_nonzero and scale == 0.0:
        raise EmbeddingVectorError("embedding vector must not be zero")
    if not require_nonzero and scale == 0.0:
        return vector
    return vector


def l2_normalize(
    values: Sequence[float],
    *,
    expected_dimensions: int | None = None,
) -> tuple[float, ...]:
    """Return a finite unit-length vector."""
    vector = validate_vector(values, expected_dimensions=expected_dimensions)
    scale = max(abs(value) for value in vector)
    scaled = tuple(value / scale for value in vector)
    scaled_norm = math.sqrt(math.fsum(value * value for value in scaled))
    if not math.isfinite(scaled_norm) or scaled_norm <= 0.0:
        raise EmbeddingVectorError("embedding norm is not finite")
    normalized = tuple(value / scale / scaled_norm for value in vector)
    return validate_vector(normalized, expected_dimensions=len(vector))


def _pack_finite_float32(values: Sequence[float]) -> bytes:
    try:
        blob = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as exc:
        raise EmbeddingVectorError(
            "embedding cannot be represented as float32"
        ) from exc
    unpacked = struct.unpack(f"<{len(values)}f", blob)
    if not all(math.isfinite(value) for value in unpacked):
        raise EmbeddingVectorError("embedding becomes non-finite in float32")
    return blob


def pack_float32_le(
    values: Sequence[float],
    *,
    expected_dimensions: int | None = None,
    normalize: bool = True,
) -> bytes:
    """Encode a validated vector as canonical little-endian float32."""
    vector = (
        l2_normalize(values, expected_dimensions=expected_dimensions)
        if normalize
        else validate_vector(
            values,
            expected_dimensions=expected_dimensions,
            require_nonzero=True,
        )
    )
    first_blob = _pack_finite_float32(vector)
    if not normalize:
        return first_blob
    float32_values = struct.unpack(f"<{len(vector)}f", first_blob)
    normalized32 = l2_normalize(
        float32_values,
        expected_dimensions=len(vector),
    )
    return _pack_finite_float32(normalized32)


def unpack_float32_le(
    blob: bytes,
    *,
    dimensions: int,
    normalize: bool = False,
) -> tuple[float, ...]:
    """Decode a canonical float32-le BLOB."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise EmbeddingVectorError("embedding blob must be bytes-like")
    raw = bytes(blob)
    expected_size = dimensions * _FLOAT32_SIZE
    if len(raw) != expected_size:
        raise EmbeddingVectorError(
            f"embedding blob length {len(raw)} does not match "
            f"{dimensions} float32 values ({expected_size} bytes)"
        )
    try:
        vector = struct.unpack(f"<{dimensions}f", raw)
    except struct.error as exc:
        raise EmbeddingVectorError("embedding blob could not be decoded") from exc
    validated = validate_vector(vector, expected_dimensions=dimensions)
    return l2_normalize(validated, expected_dimensions=dimensions) if normalize else validated


def dot_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return dot product for finite, nonzero vectors."""
    lhs = validate_vector(left)
    rhs = validate_vector(right, expected_dimensions=len(lhs))
    score = math.fsum(a * b for a, b in zip(lhs, rhs, strict=True))
    if not math.isfinite(score):
        raise EmbeddingVectorError("embedding similarity is not finite")
    return max(-1.0, min(1.0, score))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Normalize both inputs and calculate cosine similarity."""
    lhs = l2_normalize(left)
    rhs = l2_normalize(right, expected_dimensions=len(lhs))
    return dot_similarity(lhs, rhs)
