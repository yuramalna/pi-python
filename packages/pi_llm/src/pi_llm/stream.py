"""Top-level streaming API.

The four functions here are the primary interface for making LLM calls.
Use ``stream_simple`` / ``complete_simple`` for most use cases.
Use ``stream`` / ``complete`` when you need provider-specific options.
"""

from __future__ import annotations

from pi_llm.api_registry import get_api_provider
from pi_llm.types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream


def _resolve_api_provider(api: str):
    """Resolve API provider or raise."""
    provider = get_api_provider(api)
    if not provider:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response using provider-specific options.

    Args:
        model: The model to use.
        context: Conversation context (system prompt, messages, tools).
        options: Provider-specific streaming options.

    Returns:
        An async iterable event stream. Use ``async for event in stream``
        to consume events, or ``await stream.result()`` for the final message.

    Raises:
        ValueError: If no provider is registered for the model's API.
    """
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Non-streaming convenience wrapper around ``stream()``.

    Args:
        model: The model to use.
        context: Conversation context.
        options: Provider-specific streaming options.

    Returns:
        The completed assistant message.
    """
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream a response with automatic reasoning support.

    This is the recommended entry point for most use cases. It handles
    reasoning level mapping and option building automatically.

    Args:
        model: The model to use.
        context: Conversation context (system prompt, messages, tools).
        options: Simplified options including reasoning level.

    Returns:
        An async iterable event stream.

    Raises:
        ValueError: If no provider is registered for the model's API.
    """
    provider = _resolve_api_provider(model.api)
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Non-streaming convenience wrapper around ``stream_simple()``.

    Args:
        model: The model to use.
        context: Conversation context.
        options: Simplified options including reasoning level.

    Returns:
        The completed assistant message.
    """
    s = stream_simple(model, context, options)
    return await s.result()
