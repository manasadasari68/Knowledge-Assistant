"""Configurable OpenAI-compatible text embedding client."""

import os
from typing import Any

import httpx

from app.services.openrouter import (
    OpenRouterConfigurationError,
    get_openrouter_config,
)


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding provider cannot return vectors."""


class EmbeddingService:
    def __init__(self) -> None:
        try:
            config = get_openrouter_config()
        except OpenRouterConfigurationError as exc:
            raise EmbeddingError(str(exc)) from exc
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        try:
            self.dimensions = int(dimensions) if dimensions else None
        except ValueError as exc:
            raise EmbeddingError("EMBEDDING_DIMENSIONS must be a positive integer") from exc
        if self.dimensions is not None and self.dimensions <= 0:
            raise EmbeddingError("EMBEDDING_DIMENSIONS must be a positive integer")
        if not self.model:
            raise EmbeddingError("EMBEDDING_MODEL must not be empty")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=90.0,
            )
            response.raise_for_status()
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            detail = "embedding provider request failed"
            if isinstance(exc, httpx.HTTPStatusError):
                detail = f"embedding provider returned HTTP {exc.response.status_code}"
            raise EmbeddingError(detail) from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        if not vectors or not vectors[0]:
            raise EmbeddingError("embedding provider returned an empty vector")
        expected_dimensions = self.dimensions or len(vectors[0])
        for vector in vectors:
            if len(vector) != expected_dimensions:
                raise EmbeddingError("embedding provider returned inconsistent vector dimensions")
        return vectors
