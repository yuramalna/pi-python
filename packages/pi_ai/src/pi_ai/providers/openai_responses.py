"""OpenAI Responses API provider.

Implements the streaming interface for OpenAI's Responses API.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from pi_ai.env_api_keys import get_env_api_key
from pi_ai.models import supports_xhigh
from pi_ai.providers.openai_responses_shared import (
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pi_ai.providers.simple_options import build_base_options, clamp_reasoning
from pi_ai.types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream

OPENAI_TOOL_CALL_PROVIDERS: set[str] = {"openai", "openai-codex", "opencode"}


@dataclass
class OpenAIResponsesOptions(StreamOptions):
    """Extended options for the OpenAI Responses API.

    Attributes:
        reasoning_effort: Reasoning effort level (e.g. ``"medium"``).
        reasoning_summary: Reasoning summary mode (e.g. ``"auto"``).
        service_tier: Service tier (e.g. ``"flex"``, ``"priority"``).
    """

    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    service_tier: str | None = None


# =============================================================================
# Provider entry points
# =============================================================================


def stream_openai_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream from the OpenAI Responses API.

    Returns an event stream immediately and processes the API response
    in a background asyncio task.

    Args:
        model: Model to use for the request.
        context: Conversation context.
        options: OpenAI-specific options.

    Returns:
        An async iterable event stream of ``AssistantMessageEvent``.
    """
    stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
            client = _create_client(model, api_key, options.headers if options else None)
            params = _build_params(model, context, options)

            # on_payload hook (mirrors TS lines 93-96)
            if options and options.on_payload:
                next_params = options.on_payload(params, model)
                if next_params is not None:
                    params = next_params

            openai_stream = await client.responses.create(**params)
            stream.push(StartEvent(partial=output))

            await process_responses_stream(
                openai_stream, output, stream, model,
                OpenAIResponsesStreamOptions(
                    service_tier=options.service_tier if options else None,
                    apply_service_tier_pricing=_apply_service_tier_pricing,
                ),
            )

            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            stream.push(DoneEvent(reason=output.stop_reason, message=output))

        except asyncio.CancelledError:
            output.stop_reason = "aborted"
            output.error_message = "Request was aborted"
            stream.push(ErrorEvent(reason="aborted", error=output))

        except Exception as e:
            output.stop_reason = "error"
            output.error_message = str(e)
            stream.push(ErrorEvent(reason="error", error=output))

        finally:
            stream.end()

    asyncio.get_running_loop().create_task(_run())
    return stream


def stream_simple_openai_responses(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Simplified entry point with automatic option building.

    Validates the API key, maps ``SimpleStreamOptions`` to
    ``OpenAIResponsesOptions``, and delegates to ``stream_openai_responses``.
    """
    api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
    if not api_key:
        raise ValueError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, options, api_key)
    reasoning_effort = (
        (options.reasoning if options else None)
        if supports_xhigh(model)
        else clamp_reasoning(options.reasoning if options else None)
    )

    return stream_openai_responses(
        model,
        context,
        OpenAIResponsesOptions(
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            api_key=base.api_key,
            cancel_event=base.cancel_event,
            cache_retention=base.cache_retention,
            session_id=base.session_id,
            headers=base.headers,
            on_payload=base.on_payload,
            max_retry_delay_ms=base.max_retry_delay_ms,
            metadata=base.metadata,
            reasoning_effort=reasoning_effort,
        ),
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _create_client(
    model: Model,
    api_key: str,
    options_headers: dict[str, str] | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with auth and headers."""
    if not api_key:
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument."
            )
        api_key = env_key

    headers = dict(model.headers) if model.headers else {}
    # Skip GitHub Copilot header logic (not needed for this port)
    if options_headers:
        headers.update(options_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers,
    )


def _build_params(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> dict:
    """Build OpenAI API request parameters from context and options."""
    messages = convert_responses_messages(model, context, OPENAI_TOOL_CALL_PROVIDERS)

    cache_retention = _resolve_cache_retention(options.cache_retention if options else None)
    params: dict = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    # Prompt caching (TS lines 195-196)
    if cache_retention != "none" and options and options.session_id:
        params["prompt_cache_key"] = options.session_id
    prompt_cache_ret = _get_prompt_cache_retention(model.base_url, cache_retention)
    if prompt_cache_ret:
        params["prompt_cache_retention"] = prompt_cache_ret

    if options and options.max_tokens:
        params["max_output_tokens"] = options.max_tokens
    if options and options.temperature is not None:
        params["temperature"] = options.temperature
    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if context.tools:
        params["tools"] = convert_responses_tools(context.tools)

    # Reasoning config (TS lines 216-226)
    if model.reasoning:
        if options and (options.reasoning_effort or options.reasoning_summary):
            params["reasoning"] = {
                "effort": options.reasoning_effort or "medium",
                "summary": options.reasoning_summary or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        else:
            params["reasoning"] = {"effort": "none"}

    return params


def _resolve_cache_retention(
    cache_retention: CacheRetention | None = None,
) -> CacheRetention:
    """Resolve cache retention preference from arg or env var."""
    if cache_retention:
        return cache_retention
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


def _get_prompt_cache_retention(
    base_url: str, cache_retention: CacheRetention
) -> str | None:
    """Get ``prompt_cache_retention`` value for OpenAI direct API calls."""
    if cache_retention != "long":
        return None
    if "api.openai.com" in base_url:
        return "24h"
    return None


def _get_service_tier_cost_multiplier(service_tier: str | None) -> float:
    """Get cost multiplier for service tier."""
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.0
    return 1.0


def _apply_service_tier_pricing(usage: Usage, service_tier: str | None) -> None:
    """Apply service tier cost multiplier to usage."""
    multiplier = _get_service_tier_cost_multiplier(service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )
