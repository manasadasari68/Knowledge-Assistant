"""Shared OpenRouter credentials and endpoint configuration."""

import os
from dataclasses import dataclass


class OpenRouterConfigurationError(RuntimeError):
    """Raised when OpenRouter credentials are not configured."""


@dataclass(frozen=True)
class OpenRouterConfig:
    base_url: str
    api_key: str


def get_openrouter_config() -> OpenRouterConfig:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterConfigurationError(
            "Set EMBEDDING_API_KEY (or OPENROUTER_API_KEY) in app/api/.env"
        )
    base_url = os.getenv(
        "EMBEDDING_API_BASE_URL", "https://openrouter.ai/api/v1"
    ).rstrip("/")
    return OpenRouterConfig(base_url=base_url, api_key=api_key)
