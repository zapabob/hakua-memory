"""Concrete OpenAI-compatible embedding adapter for operator-managed llama.cpp."""

from __future__ import annotations

import ipaddress
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from .base import EmbeddingBackendError, EmbeddingModelIdentity
from .vectors import validate_vector


class LlamaCppEmbeddingBackend:
    """Call one configured llama.cpp ``/v1/embeddings`` endpoint only."""

    _MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        revision: str,
        dimensions: int,
        serializer_version: int,
        timeout_seconds: float,
        allow_remote: bool = False,
    ) -> None:
        self._endpoint = self._normalize_endpoint(
            endpoint,
            allow_remote=allow_remote,
        )
        self._timeout_seconds = self._positive_timeout(timeout_seconds)
        self._model = str(model or "").strip()
        if not self._model:
            raise EmbeddingBackendError("embedding model must not be empty")
        try:
            self._identity = EmbeddingModelIdentity(
                provider="llama.cpp",
                model=self._model,
                revision=str(revision or "").strip(),
                dimensions=self._positive_int(dimensions, "embedding dimensions"),
                serializer_version=self._positive_int(
                    serializer_version,
                    "embedding serializer version",
                ),
            )
        except ValueError as exc:
            raise EmbeddingBackendError("invalid llama.cpp embedding configuration") from exc

    @property
    def identity(self) -> EmbeddingModelIdentity:
        return self._identity

    def available(self) -> bool:
        """Return configuration availability without probing the server."""
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
        return self._request_embeddings(inputs)

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool):
            raise EmbeddingBackendError(f"{label} must be positive")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise EmbeddingBackendError(f"{label} must be positive") from exc
        if result <= 0:
            raise EmbeddingBackendError(f"{label} must be positive")
        return result

    @staticmethod
    def _positive_timeout(value: Any) -> float:
        if isinstance(value, bool):
            raise EmbeddingBackendError("embedding timeout must be positive")
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise EmbeddingBackendError("embedding timeout must be positive") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise EmbeddingBackendError("embedding timeout must be positive")
        return timeout

    @classmethod
    def _normalize_endpoint(cls, endpoint: str, *, allow_remote: bool) -> str:
        value = str(endpoint or "").strip()
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EmbeddingBackendError("embedding endpoint must be an absolute HTTP URL")
        if parsed.username or parsed.password:
            raise EmbeddingBackendError("embedding endpoint must not include credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise EmbeddingBackendError("embedding endpoint must not include a path or query")
        host = parsed.hostname
        if not host:
            raise EmbeddingBackendError("embedding endpoint must include a host")
        if not allow_remote and not cls._is_loopback_host(host):
            raise EmbeddingBackendError(
                "embedding endpoint must be loopback unless allow_remote is enabled"
            )
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host.casefold() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps(
            {
                "model": self._model,
                "input": inputs,
                "encoding_format": "float",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._endpoint}/v1/embeddings",
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                status = response.status if response.status is not None else response.getcode()
                if int(status) != 200:
                    raise EmbeddingBackendError("embedding request returned an unexpected status")
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
        except EmbeddingBackendError:
            raise
        except urllib.error.HTTPError as exc:
            raise EmbeddingBackendError(
                f"embedding request failed with HTTP status {exc.code}"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise EmbeddingBackendError(
                f"embedding request failed: {type(exc).__name__}"
            ) from exc

        if len(raw) > self._MAX_RESPONSE_BYTES:
            raise EmbeddingBackendError("embedding response is too large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingBackendError("embedding response is not valid JSON") from exc
        return self._parse_response(payload, expected_count=len(inputs))

    def _parse_response(self, payload: Any, *, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, Mapping):
            raise EmbeddingBackendError("embedding response must be an object")
        response_model = payload.get("model")
        if (
            isinstance(response_model, str)
            and response_model.strip()
            and response_model.strip() != self._model
        ):
            raise EmbeddingBackendError(
                "embedding response model alias does not match the configured model"
            )

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise EmbeddingBackendError("embedding response count does not match input count")

        vectors: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, Mapping):
                raise EmbeddingBackendError("embedding response item is malformed")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index in vectors:
                raise EmbeddingBackendError("embedding response indices are invalid")
            values = item.get("embedding")
            if not isinstance(values, list):
                raise EmbeddingBackendError("embedding response vector is malformed")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            ):
                raise EmbeddingBackendError("embedding contains a non-numeric value")
            try:
                vector = validate_vector(
                    values,
                    expected_dimensions=self._identity.dimensions,
                )
            except EmbeddingBackendError:
                raise
            except (TypeError, ValueError) as exc:
                raise EmbeddingBackendError("embedding response vector is invalid") from exc
            vectors[index] = list(vector)

        if set(vectors) != set(range(expected_count)):
            raise EmbeddingBackendError("embedding response indices do not cover all inputs")
        return [list(vectors[index]) for index in range(expected_count)]


__all__ = ["LlamaCppEmbeddingBackend"]
