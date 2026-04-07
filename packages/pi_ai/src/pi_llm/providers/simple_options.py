"""Option building helpers for providers.

Maps ``SimpleStreamOptions`` to provider-specific options, handles reasoning
level clamping, and adjusts token budgets for thinking-enabled models.
"""

from __future__ import annotations

from pi_llm.types import Model, SimpleStreamOptions, StreamOptions, ThinkingLevel


def build_base_options(
    model: Model,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    """Map ``SimpleStreamOptions`` to base ``StreamOptions``.

    Args:
        model: The model (used for default max_tokens).
        options: Simplified options to convert.
        api_key: API key override.

    Returns:
        A ``StreamOptions`` instance.
    """
    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=(options.max_tokens if options else None) or min(model.max_tokens, 32000),
        cancel_event=options.cancel_event if options else None,
        api_key=api_key or (options.api_key if options else None),
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> ThinkingLevel | None:
    """Clamp ``"xhigh"`` to ``"high"`` for models that don't support it."""
    return "high" if effort == "xhigh" else effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Calculate token limits for thinking-enabled models.

    Returns:
        A ``(max_tokens, thinking_budget)`` tuple.
    """
    default_budgets = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
    budgets = {**default_budgets, **(custom_budgets or {})}

    min_output_tokens = 1024
    level = clamp_reasoning(reasoning_level) or "medium"
    thinking_budget = budgets.get(level, 8192)
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget
