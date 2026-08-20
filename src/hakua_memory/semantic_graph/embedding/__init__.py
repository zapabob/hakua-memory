"""Embedding contracts and adapters for Semantic Graph retrieval."""

from .base import (
    EmbeddingBackend,
    EmbeddingBackendError,
    EmbeddingModelIdentity,
)
from .fake import DeterministicFakeEmbeddingBackend
from .llama_cpp import LlamaCppEmbeddingBackend
from .serializer import (
    QUERY_INSTRUCTION,
    serialize_embedding_node,
    serialize_embedding_query,
    source_text_hash,
)
from .vectors import (
    FLOAT32_LE_DTYPE,
    EmbeddingVectorError,
    cosine_similarity,
    dot_similarity,
    l2_normalize,
    pack_float32_le,
    unpack_float32_le,
    validate_vector,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendError",
    "EmbeddingModelIdentity",
    "DeterministicFakeEmbeddingBackend",
    "LlamaCppEmbeddingBackend",
    "QUERY_INSTRUCTION",
    "serialize_embedding_node",
    "serialize_embedding_query",
    "source_text_hash",
    "FLOAT32_LE_DTYPE",
    "EmbeddingVectorError",
    "cosine_similarity",
    "dot_similarity",
    "l2_normalize",
    "pack_float32_le",
    "unpack_float32_le",
    "validate_vector",
]
