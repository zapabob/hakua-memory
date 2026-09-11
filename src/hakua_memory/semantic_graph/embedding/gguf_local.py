"""In-process llama-cpp-python embedding backend for local GGUF models."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import EmbeddingBackendError, EmbeddingModelIdentity
from .vectors import validate_vector


class LlamaCppPythonEmbeddingBackend:
    """Load a local GGUF embedding model via ``llama-cpp-python``.

    Intended for models such as BGE-M3 GGUF where an HTTP server is optional.
    Requires the ``[embedding]`` extra (``llama-cpp-python``).
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        dimensions: int = 1024,
        model: str = "nsfw-bge-m3-v5-q6_k",
        revision: str = "q6_k",
        serializer_version: int = 1,
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        provider: str = "llama-cpp-python",
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise EmbeddingBackendError(f"embedding model file not found: {path}")
        try:
            self._identity = EmbeddingModelIdentity(
                provider=provider,
                model=str(model).strip() or path.stem,
                revision=str(revision or "").strip(),
                dimensions=int(dimensions),
                serializer_version=int(serializer_version),
            )
        except ValueError as exc:
            raise EmbeddingBackendError("invalid GGUF embedding configuration") from exc

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise EmbeddingBackendError(
                "llama-cpp-python is required; install hakua-memory[embedding]"
            ) from exc

        self._lock = threading.RLock()
        self._llm = Llama(
            model_path=str(path),
            embedding=True,
            n_ctx=max(64, int(n_ctx)),
            n_gpu_layers=int(n_gpu_layers),
            n_batch=max(1, int(n_batch)),
            verbose=False,
        )
        self._model_path = path

    @property
    def identity(self) -> EmbeddingModelIdentity:
        return self._identity

    @property
    def model_path(self) -> Path:
        return self._model_path

    def available(self) -> bool:
        return True

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingBackendError("embedding inputs must be a sequence of strings")
        inputs = list(texts)
        if not inputs:
            return []
        if any(not isinstance(text, str) for text in inputs):
            raise EmbeddingBackendError("embedding inputs must contain only strings")

        vectors: list[list[float]] = []
        with self._lock:
            for text in inputs:
                try:
                    payload: dict[str, Any] = self._llm.create_embedding(text)
                except Exception as exc:  # noqa: BLE001 - surface as backend error
                    raise EmbeddingBackendError(
                        f"GGUF embedding failed: {type(exc).__name__}"
                    ) from exc
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list) or not data:
                    raise EmbeddingBackendError("GGUF embedding response is empty")
                values = data[0].get("embedding") if isinstance(data[0], dict) else None
                if not isinstance(values, list):
                    raise EmbeddingBackendError("GGUF embedding vector is malformed")
                try:
                    vectors.append(
                        list(
                            validate_vector(
                                values,
                                expected_dimensions=self._identity.dimensions,
                            )
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    raise EmbeddingBackendError("GGUF embedding vector is invalid") from exc
        return vectors

    def close(self) -> None:
        """Release the underlying llama.cpp context if supported."""
        with self._lock:
            closer = getattr(self._llm, "close", None)
            if callable(closer):
                closer()


__all__ = ["LlamaCppPythonEmbeddingBackend"]
