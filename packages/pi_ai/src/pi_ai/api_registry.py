"""API provider registry.

Module-level registry mapping API name strings to provider implementations.
Providers are registered at startup via ``register_builtin_providers()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pi_ai.types import Context, Model, SimpleStreamOptions, StreamOptions
from pi_ai.utils.event_stream import AssistantMessageEventStream

StreamFn = Callable[[Model, Context, StreamOptions | None], AssistantMessageEventStream]
StreamSimpleFn = Callable[[Model, Context, SimpleStreamOptions | None], AssistantMessageEventStream]


@dataclass
class _InternalProvider:
    api: str
    stream: StreamFn
    stream_simple: StreamSimpleFn


@dataclass
class _RegisteredProvider:
    provider: _InternalProvider
    source_id: str | None = None


_registry: dict[str, _RegisteredProvider] = {}


def _wrap_stream(api: str, stream_fn: StreamFn) -> StreamFn:
    """Wrap a stream function with an API mismatch check."""

    def wrapped(
        model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return stream_fn(model, context, options)

    return wrapped


def _wrap_stream_simple(api: str, stream_fn: StreamSimpleFn) -> StreamSimpleFn:
    """Wrap a stream_simple function with an API mismatch check."""

    def wrapped(
        model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return stream_fn(model, context, options)

    return wrapped


def register_api_provider(
    api: str,
    stream: StreamFn,
    stream_simple: StreamSimpleFn,
    source_id: str | None = None,
) -> None:
    """Register an API provider with stream implementations.

    Args:
        api: API name (e.g. ``"openai-responses"``).
        stream: Provider-specific stream function.
        stream_simple: Simplified stream function with reasoning support.
        source_id: Optional identifier for bulk unregistration.
    """
    _registry[api] = _RegisteredProvider(
        provider=_InternalProvider(
            api=api,
            stream=_wrap_stream(api, stream),
            stream_simple=_wrap_stream_simple(api, stream_simple),
        ),
        source_id=source_id,
    )


def get_api_provider(api: str) -> _InternalProvider | None:
    """Look up a registered provider by API name."""
    entry = _registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[_InternalProvider]:
    """List all registered providers."""
    return [e.provider for e in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove all providers registered with the given source ID."""
    to_remove = [api for api, e in _registry.items() if e.source_id == source_id]
    for api in to_remove:
        del _registry[api]


def clear_api_providers() -> None:
    """Remove all registered providers."""
    _registry.clear()
