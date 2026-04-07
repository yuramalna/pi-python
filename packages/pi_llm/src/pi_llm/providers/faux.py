"""Deterministic mock LLM provider for testing.

Provides ``register_faux_provider()`` which registers a fake API provider
that returns scripted responses, simulating streaming events without
hitting any real LLM API. Useful for unit and integration tests.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pi_llm.api_registry import register_api_provider, unregister_api_providers
from pi_llm.types import (
    AssistantMessage,
    Context,
    CostBreakdown,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pi_llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5

DEFAULT_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
    cost=CostBreakdown(),
)

# ---------------------------------------------------------------------------
# Public content factories
# ---------------------------------------------------------------------------


def faux_text(text: str) -> TextContent:
    """Create a ``TextContent`` block for scripted responses."""
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    """Create a ``ThinkingContent`` block for scripted responses."""
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    id: str | None = None,
) -> ToolCall:
    """Create a ``ToolCall`` block for scripted responses."""
    return ToolCall(
        id=id or _random_id("tool"),
        name=name,
        arguments=arguments,
    )


FauxContentBlock = TextContent | ThinkingContent | ToolCall
"""Content block types that can appear in a faux assistant message."""

FauxResponseFactory = Callable[
    [Context, StreamOptions | None, dict[str, int], Model],
    "AssistantMessage",
]
"""Callable ``(context, options, state, model) -> AssistantMessage``
that builds a response dynamically. Can also be async."""

FauxResponseStep = AssistantMessage | FauxResponseFactory
"""A scripted response: either a static message or a factory function."""


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------


@dataclass
class FauxModelDefinition:
    """Definition for a mock model."""

    id: str
    name: str | None = None
    reasoning: bool = False
    input_types: list[str] = field(default_factory=lambda: ["text", "image"])
    cost: ModelCost | None = None
    context_window: int = 128_000
    max_tokens: int = 16_384


@dataclass
class RegisterFauxProviderOptions:
    """Options for ``register_faux_provider()``."""

    api: str | None = None
    provider: str | None = None
    models: list[FauxModelDefinition] | None = None
    tokens_per_second: float | None = None
    token_size_min: int | None = None
    token_size_max: int | None = None


# ---------------------------------------------------------------------------
# Registration result
# ---------------------------------------------------------------------------


class FauxProviderRegistration:
    """Handle returned by ``register_faux_provider()``.

    Use this to configure scripted responses and inspect call state.

    Attributes:
        api: The API name this provider was registered under.
        models: List of mock ``Model`` instances.
        state: Dict with ``callCount`` tracking total stream calls.
    """

    def __init__(
        self,
        api: str,
        models: list[Model],
        state: dict[str, int],
        *,
        _set_responses: Callable[[list[FauxResponseStep]], None],
        _append_responses: Callable[[list[FauxResponseStep]], None],
        _get_pending: Callable[[], int],
        _unregister: Callable[[], None],
    ) -> None:
        self.api = api
        self.models = models
        self.state = state
        self._set_responses = _set_responses
        self._append_responses = _append_responses
        self._get_pending = _get_pending
        self._unregister = _unregister

    def get_model(self, model_id: str | None = None) -> Model | None:
        """Get a mock model by ID, or the first model if no ID given."""
        if model_id is None:
            return self.models[0]
        return next((m for m in self.models if m.id == model_id), None)

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        """Replace the scripted response queue."""
        self._set_responses(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        """Add responses to the end of the queue."""
        self._append_responses(responses)

    def get_pending_response_count(self) -> int:
        """Return the number of queued responses remaining."""
        return self._get_pending()

    def unregister(self) -> None:
        """Remove this provider from the registry."""
        self._unregister()


# ---------------------------------------------------------------------------
# Public factory: faux_assistant_message
# ---------------------------------------------------------------------------


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    """Build a complete ``AssistantMessage`` for scripted responses.

    Args:
        content: Text string, single content block, or list of blocks.
        stop_reason: Stop reason (default ``"stop"``).
        error_message: Error description if stop_reason is ``"error"``.
        response_id: Provider-assigned response ID.
        timestamp: Unix timestamp in ms (defaults to current time).
    """
    blocks = _normalize_content(content)
    return AssistantMessage(
        content=blocks,
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=DEFAULT_USAGE.model_copy(),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp or _now_ms(),
    )


# ---------------------------------------------------------------------------
# Main registration function
# ---------------------------------------------------------------------------


def register_faux_provider(
    options: RegisterFauxProviderOptions | None = None,
) -> FauxProviderRegistration:
    """Register a deterministic mock LLM provider.

    Returns a ``FauxProviderRegistration`` that lets you queue scripted
    ``AssistantMessage`` responses. Each call to the registered stream
    function pops the next response and simulates streaming events.

    Args:
        options: Configuration (API name, models, token speed, etc.).

    Returns:
        A registration handle for managing responses and models.
    """
    opts = options or RegisterFauxProviderOptions()
    api = opts.api or _random_id(DEFAULT_API)
    provider = opts.provider or DEFAULT_PROVIDER
    source_id = _random_id("faux-provider")
    min_token_size = max(
        1,
        min(
            opts.token_size_min or DEFAULT_MIN_TOKEN_SIZE,
            opts.token_size_max or DEFAULT_MAX_TOKEN_SIZE,
        ),
    )
    max_token_size = max(
        min_token_size, opts.token_size_max or DEFAULT_MAX_TOKEN_SIZE
    )
    tokens_per_second = opts.tokens_per_second

    pending_responses: list[FauxResponseStep] = []
    state: dict[str, int] = {"callCount": 0}
    prompt_cache: dict[str, str] = {}

    # Build models
    model_defs = opts.models or [FauxModelDefinition(id=DEFAULT_MODEL_ID, name=DEFAULT_MODEL_NAME)]
    models: list[Model] = []
    for defn in model_defs:
        models.append(
            Model(
                id=defn.id,
                name=defn.name or defn.id,
                api=api,
                provider=provider,
                base_url=DEFAULT_BASE_URL,
                reasoning=defn.reasoning,
                input_types=defn.input_types,
                cost=defn.cost,
                context_window=defn.context_window,
                max_tokens=defn.max_tokens,
            )
        )

    def stream_fn(
        request_model: Model,
        context: Context,
        stream_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = create_assistant_message_event_stream()
        step = pending_responses.pop(0) if pending_responses else None
        state["callCount"] += 1

        async def _run() -> None:
            try:
                if step is None:
                    message = _create_error_message(
                        "No more faux responses queued",
                        api, provider, request_model.id,
                    )
                    message = _with_usage_estimate(
                        message, context, stream_options, prompt_cache
                    )
                    outer.push(ErrorEvent(reason="error", error=message))
                    outer.end(message)
                    return

                if callable(step):
                    resolved = step(context, stream_options, state, request_model)
                    if asyncio.iscoroutine(resolved):
                        resolved = await resolved
                else:
                    resolved = step

                message = _clone_message(resolved, api, provider, request_model.id)
                message = _with_usage_estimate(
                    message, context, stream_options, prompt_cache
                )
                await _stream_with_deltas(
                    outer, message,
                    min_token_size, max_token_size,
                    tokens_per_second,
                    stream_options,
                )
            except Exception as exc:
                message = _create_error_message(
                    str(exc), api, provider, request_model.id,
                )
                outer.push(ErrorEvent(reason="error", error=message))
                outer.end(message)

        asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(_run()))
        return outer

    def stream_simple_fn(
        model: Model,
        context: Context,
        simple_options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return stream_fn(model, context, simple_options)

    register_api_provider(api, stream_fn, stream_simple_fn, source_id)

    return FauxProviderRegistration(
        api=api,
        models=models,
        state=state,
        _set_responses=lambda r: _replace_list(pending_responses, r),
        _append_responses=lambda r: pending_responses.extend(r),
        _get_pending=lambda: len(pending_responses),
        _unregister=lambda: unregister_api_providers(source_id),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_list(target: list, source: list) -> None:
    target.clear()
    target.extend(source)


def _random_id(prefix: str) -> str:
    return f"{prefix}:{int(time.time() * 1000)}:{random.randint(0, 2**32):x}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _normalize_content(
    content: str | FauxContentBlock | list[FauxContentBlock],
) -> list[TextContent | ThinkingContent | ToolCall]:
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return list(content)
    return [content]


def _content_to_text(
    content: str | list[TextContent | ImageContent],
) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(
    content: list[TextContent | ThinkingContent | ToolCall],
) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ThinkingContent):
            parts.append(block.thinking)
        elif isinstance(block, ToolCall):
            parts.append(f"{block.name}:{json.dumps(block.arguments)}")
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    if hasattr(message, "role"):
        if message.role == "user":
            return _content_to_text(message.content)
        if message.role == "assistant":
            return _assistant_content_to_text(message.content)
        if message.role == "toolResult":
            parts = [message.tool_name]
            parts.extend(_content_to_text([b]) for b in message.content)
            return "\n".join(parts)
    return ""


def _serialize_context(context: Context) -> str:
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for msg in context.messages:
        role = msg.role if hasattr(msg, "role") else "unknown"
        parts.append(f"{role}:{_message_to_text(msg)}")
    if context.tools:
        parts.append(f"tools:{json.dumps([t.model_dump() for t in context.tools])}")
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    idx = 0
    while idx < length and a[idx] == b[idx]:
        idx += 1
    return idx


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(
        _assistant_content_to_text(message.content)
    )
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0

    session_id = getattr(options, "session_id", None) if options else None
    cache_retention = getattr(options, "cache_retention", None) if options else None

    if session_id and cache_retention != "none":
        previous_prompt = prompt_cache.get(session_id)
        if previous_prompt:
            cached_chars = _common_prefix_length(previous_prompt, prompt_text)
            cache_read = _estimate_tokens(previous_prompt[:cached_chars])
            cache_write = _estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    return message.model_copy(
        update={
            "usage": Usage(
                input=input_tokens,
                output=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
                total_tokens=input_tokens + output_tokens + cache_read + cache_write,
                cost=CostBreakdown(),
            ),
        }
    )


def _split_by_token_size(
    text: str, min_size: int, max_size: int
) -> list[str]:
    chunks: list[str] = []
    idx = 0
    while idx < len(text):
        token_size = min_size + random.randint(0, max_size - min_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[idx : idx + char_size])
        idx += char_size
    return chunks or [""]


def _clone_message(
    message: AssistantMessage,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    cloned = message.model_copy(deep=True)
    return cloned.model_copy(
        update={
            "api": api,
            "provider": provider,
            "model": model_id,
            "timestamp": cloned.timestamp or _now_ms(),
            "usage": cloned.usage if cloned.usage else DEFAULT_USAGE.model_copy(),
        }
    )


def _create_error_message(
    error_text: str,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=DEFAULT_USAGE.model_copy(),
        stop_reason="error",
        error_message=error_text,
        timestamp=_now_ms(),
    )


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    return partial.model_copy(
        update={
            "stop_reason": "aborted",
            "error_message": "Request was aborted",
            "timestamp": _now_ms(),
        }
    )


def _is_cancelled(options: StreamOptions | None) -> bool:
    if options is None:
        return False
    cancel = getattr(options, "cancel_event", None)
    return cancel.is_set() if cancel else False


async def _schedule_chunk(
    chunk: str, tokens_per_second: float | None
) -> None:
    if not tokens_per_second or tokens_per_second <= 0:
        await asyncio.sleep(0)
        return
    delay = (_estimate_tokens(chunk) / tokens_per_second)
    await asyncio.sleep(delay)


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
    options: StreamOptions | None,
) -> None:
    # Build a mutable partial message
    partial = message.model_copy(update={"content": []})

    if _is_cancelled(options):
        aborted = _create_aborted_message(partial)
        stream.push(ErrorEvent(reason="aborted", error=aborted))
        stream.end(aborted)
        return

    stream.push(StartEvent(partial=partial.model_copy()))

    for index, block in enumerate(message.content):
        if _is_cancelled(options):
            aborted = _create_aborted_message(partial)
            stream.push(ErrorEvent(reason="aborted", error=aborted))
            stream.end(aborted)
            return

        if isinstance(block, ThinkingContent):
            # Add empty thinking block to partial
            partial = partial.model_copy(
                update={"content": [*partial.content, ThinkingContent(thinking="")]}
            )
            stream.push(ThinkingStartEvent(
                content_index=index, partial=partial.model_copy(),
            ))
            for chunk in _split_by_token_size(block.thinking, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if _is_cancelled(options):
                    aborted = _create_aborted_message(partial)
                    stream.push(ErrorEvent(reason="aborted", error=aborted))
                    stream.end(aborted)
                    return
                # Update thinking text in partial
                current_thinking = partial.content[index]
                updated = ThinkingContent(thinking=current_thinking.thinking + chunk)
                new_content = list(partial.content)
                new_content[index] = updated
                partial = partial.model_copy(update={"content": new_content})
                stream.push(ThinkingDeltaEvent(
                    content_index=index, delta=chunk, partial=partial.model_copy(),
                ))
            stream.push(ThinkingEndEvent(
                content_index=index, content=block.thinking, partial=partial.model_copy(),
            ))

        elif isinstance(block, TextContent):
            partial = partial.model_copy(
                update={"content": [*partial.content, TextContent(text="")]}
            )
            stream.push(TextStartEvent(
                content_index=index, partial=partial.model_copy(),
            ))
            for chunk in _split_by_token_size(block.text, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if _is_cancelled(options):
                    aborted = _create_aborted_message(partial)
                    stream.push(ErrorEvent(reason="aborted", error=aborted))
                    stream.end(aborted)
                    return
                current_text = partial.content[index]
                updated = TextContent(text=current_text.text + chunk)
                new_content = list(partial.content)
                new_content[index] = updated
                partial = partial.model_copy(update={"content": new_content})
                stream.push(TextDeltaEvent(
                    content_index=index, delta=chunk, partial=partial.model_copy(),
                ))
            stream.push(TextEndEvent(
                content_index=index, content=block.text, partial=partial.model_copy(),
            ))

        elif isinstance(block, ToolCall):
            partial = partial.model_copy(
                update={
                    "content": [
                        *partial.content,
                        ToolCall(id=block.id, name=block.name, arguments={}),
                    ]
                }
            )
            stream.push(ToolCallStartEvent(
                content_index=index, partial=partial.model_copy(),
            ))
            args_json = json.dumps(block.arguments)
            for chunk in _split_by_token_size(args_json, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if _is_cancelled(options):
                    aborted = _create_aborted_message(partial)
                    stream.push(ErrorEvent(reason="aborted", error=aborted))
                    stream.end(aborted)
                    return
                stream.push(ToolCallDeltaEvent(
                    content_index=index, delta=chunk, partial=partial.model_copy(),
                ))
            # Set final arguments
            new_content = list(partial.content)
            new_content[index] = block
            partial = partial.model_copy(update={"content": new_content})
            stream.push(ToolCallEndEvent(
                content_index=index, tool_call=block, partial=partial.model_copy(),
            ))

    # Final event
    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))
        stream.end(message)
        return

    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end(message)
