"""Dynamic model catalog.

Functions for constructing, fetching, and comparing LLM models.
"""

from __future__ import annotations

import os
from typing import Any

from pi_llm.model_pricing import KNOWN_METADATA, KNOWN_PRICING
from pi_llm.types import CostBreakdown, Model, Usage


def get_model(provider: str, model_id: str, **overrides: Any) -> Model:
    """Construct a Model from known metadata. Works offline.

    Unknown models get sensible defaults (128k context, 16k max_tokens,
    no pricing).

    Args:
        provider: Provider name (e.g. ``"openai"``).
        model_id: Model identifier (e.g. ``"gpt-4o"``).
        **overrides: Override any ``Model`` field (e.g. ``base_url``).

    Returns:
        A fully configured ``Model`` instance.

    Example:
        >>> model = get_model("openai", "gpt-4o")
        >>> model.context_window
        128000
    """
    meta = KNOWN_METADATA.get(model_id, {})
    pricing = KNOWN_PRICING.get(model_id)
    return Model(
        id=model_id,
        name=model_id,
        api=overrides.pop("api", "openai-responses"),
        provider=provider,
        base_url=overrides.pop("base_url", "https://api.openai.com/v1"),
        reasoning=meta.get("reasoning", False),
        input_types=meta.get("input", ["text"]),
        cost=pricing,
        context_window=meta.get("context_window", 128000),
        max_tokens=meta.get("max_tokens", 16384),
        **overrides,
    )


async def fetch_models(
    provider: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[Model]:
    """Fetch available models from the provider's API.

    Each returned model is enriched with local pricing and metadata.
    This is optional — ``get_model()`` works without calling this.

    Args:
        provider: Provider name (default ``"openai"``).
        api_key: API key (falls back to ``OPENAI_API_KEY`` env var).
        base_url: Override the API base URL.

    Returns:
        List of ``Model`` instances available from the provider.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )
    response = await client.models.list()
    return [
        get_model(
            provider,
            m.id,
            base_url=base_url or "https://api.openai.com/v1",
        )
        for m in response.data
    ]


def calculate_cost(model: Model, usage: Usage) -> None:
    """Calculate dollar cost from token usage and model pricing.

    Mutates ``usage.cost`` in-place. Does nothing if the model has no pricing.

    Args:
        model: Model with pricing information.
        usage: Token usage to calculate cost for.
    """
    if not model.cost:
        return
    usage.cost = CostBreakdown(
        input=(model.cost.input / 1_000_000) * usage.input,
        output=(model.cost.output / 1_000_000) * usage.output,
        cache_read=(model.cost.cache_read / 1_000_000) * usage.cache_read,
        cache_write=(model.cost.cache_write / 1_000_000) * usage.cache_write,
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output
        + usage.cost.cache_read + usage.cost.cache_write
    )


def supports_xhigh(model: Model) -> bool:
    """Check if a model supports the ``"xhigh"`` thinking level."""
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id or "gpt-5.4" in model.id:
        return True
    if "opus-4-6" in model.id or "opus-4.6" in model.id:
        return True
    return False


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Check if two models are the same by comparing id and provider."""
    if not a or not b:
        return False
    return a.id == b.id and a.provider == b.provider
