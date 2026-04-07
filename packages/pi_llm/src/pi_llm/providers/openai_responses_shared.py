"""Message conversion and stream processing for the OpenAI Responses API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_llm.models import calculate_cost
from pi_llm.providers.transform_messages import transform_messages
from pi_llm.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream
from pi_llm.utils.hash import short_hash
from pi_llm.utils.json_parse import parse_streaming_json
from pi_llm.utils.sanitize_unicode import sanitize_surrogates

# =============================================================================
# Utilities
# =============================================================================


def encode_text_signature_v1(id: str, phase: str | None = None) -> str:
    """Encode a TextSignatureV1 payload as JSON string (mirrors TS lines 40-44)."""
    payload: dict[str, Any] = {"v": 1, "id": id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload)


def parse_text_signature(signature: str | None) -> dict | None:
    """Parse a text signature. Returns {"id": ..., "phase": ...} or None (mirrors TS lines 46-64)."""
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
            if parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
                phase = parsed.get("phase")
                if phase in ("commentary", "final_answer"):
                    return {"id": parsed["id"], "phase": phase}
                return {"id": parsed["id"]}
        except (json.JSONDecodeError, TypeError):
            pass  # Fall through to legacy plain-string handling
    return {"id": signature}


@dataclass
class OpenAIResponsesStreamOptions:
    """Options for process_responses_stream (mirrors TS lines 66-72)."""

    service_tier: str | None = None
    apply_service_tier_pricing: Callable[[Usage, str | None], None] | None = None


# =============================================================================
# ID normalization helpers
# =============================================================================


def _normalize_id_part(part: str) -> str:
    """Replace non-alphanumeric chars, truncate to 64, strip trailing underscores (TS lines 94-98)."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", part)
    normalized = sanitized[:64]
    return normalized.rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    """Build a deterministic fc_-prefixed item ID for cross-provider calls (TS lines 100-103)."""
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64]


# =============================================================================
# Message conversion
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str],
    *,
    include_system_prompt: bool = True,
) -> list[dict]:
    """Convert pi_llm messages to OpenAI Responses API input format (mirrors TS lines 86-261)."""
    messages: list[dict] = []

    def normalize_tool_call_id(
        id: str, _target_model: Model, source: AssistantMessage
    ) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(id)
        if "|" not in id:
            return _normalize_id_part(id)
        call_id, item_id = id.split("|", 1)
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id)
            if is_foreign
            else _normalize_id_part(item_id)
        )
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(
        context.messages, model, normalize_tool_call_id
    )

    # System prompt (TS lines 121-128)
    if include_system_prompt and context.system_prompt:
        role = "developer" if model.reasoning else "system"
        messages.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}],
                })
            else:
                content = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({
                            "type": "input_text",
                            "text": sanitize_surrogates(item.text),
                        })
                    elif item.type == "image":
                        content.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{item.mime_type};base64,{item.data}",
                        })
                # Filter out images if model doesn't support them (TS lines 152-154)
                if "image" not in model.input_types:
                    content = [c for c in content if c["type"] != "input_image"]
                if not content:
                    msg_index += 1
                    continue
                messages.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            output: list[dict] = []
            assistant_msg: AssistantMessage = msg  # type: ignore[assignment]
            is_different_model = (
                assistant_msg.model != model.id
                and assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
            )

            for block in msg.content:
                if block.type == "thinking":
                    if block.thinking_signature:
                        reasoning_item = json.loads(block.thinking_signature)
                        # Strip status=null — OpenAI rejects null on input
                        # (valid values: "in_progress", "completed", "incomplete")
                        if reasoning_item.get("status") is None:
                            reasoning_item.pop("status", None)
                        output.append(reasoning_item)

                elif block.type == "text":
                    parsed_sig = parse_text_signature(block.text_signature)
                    msg_id = parsed_sig["id"] if parsed_sig else None
                    if not msg_id:
                        msg_id = f"msg_{msg_index}"
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_sig and parsed_sig.get("phase"):
                        item["phase"] = parsed_sig["phase"]
                    output.append(item)

                elif block.type == "toolCall":
                    tool_call: ToolCall = block  # type: ignore[assignment]
                    parts = tool_call.id.split("|", 1)
                    call_id = parts[0]
                    item_id: str | None = parts[1] if len(parts) > 1 else None

                    # For different-model messages, omit itemId to avoid pairing validation (TS lines 198-203)
                    if is_different_model and item_id and item_id.startswith("fc_"):
                        item_id = None

                    fc: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    }
                    if item_id is not None:
                        fc["id"] = item_id
                    output.append(fc)

            if not output:
                msg_index += 1
                continue
            messages.extend(output)

        elif msg.role == "toolResult":
            text_parts = [c.text for c in msg.content if c.type == "text"]
            text_result = "\n".join(text_parts)
            has_images = any(c.type == "image" for c in msg.content)
            has_text = len(text_result) > 0
            call_id = msg.tool_call_id.split("|", 1)[0]

            if has_images and "image" in model.input_types:
                content_parts: list[dict] = []
                if has_text:
                    content_parts.append({
                        "type": "input_text",
                        "text": sanitize_surrogates(text_result),
                    })
                for block in msg.content:
                    if block.type == "image":
                        content_parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{block.mime_type};base64,{block.data}",
                        })
                output_val: Any = content_parts
            else:
                output_val = sanitize_surrogates(text_result if has_text else "(see attached image)")

            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_val,
            })

        msg_index += 1

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(tools: list[Tool], *, strict: bool = False) -> list[dict]:
    """Convert pi_llm Tool definitions to OpenAI function tool dicts (mirrors TS lines 267-276)."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": strict,
        }
        for tool in tools
    ]


# =============================================================================
# Stream processing
# =============================================================================


def map_stop_reason(status: str | None) -> StopReason:
    """Map OpenAI response status to pi_llm StopReason (mirrors TS lines 507-526)."""
    if not status:
        return "stop"
    mapping: dict[str, StopReason] = {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "error",
        "in_progress": "stop",
        "queued": "stop",
    }
    return mapping.get(status, "stop")


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """Process OpenAI Responses API streaming events (mirrors TS lines 282-505).

    State machine tracking currentItem/currentBlock for event correlation.
    """
    # State machine
    current_item_type: str | None = None  # "reasoning" | "message" | "function_call"
    current_block: ThinkingContent | TextContent | ToolCall | None = None
    partial_json: str = ""  # Separate tracking for ToolCall partial JSON
    # Accumulation state for OpenAI item internals
    summary_parts: list[dict] = []  # For reasoning summary parts
    content_parts: list[dict] = []  # For message content parts

    blocks = output.content

    def block_index() -> int:
        return len(blocks) - 1

    async for event in openai_stream:
        event_type = event.type

        if event_type == "response.created":
            output.response_id = event.response.id

        elif event_type == "response.output_item.added":
            item = event.item
            if item.type == "reasoning":
                current_item_type = "reasoning"
                summary_parts = []
                current_block = ThinkingContent(thinking="")
                output.content.append(current_block)
                stream.push(ThinkingStartEvent(
                    content_index=block_index(), partial=output
                ))

            elif item.type == "message":
                current_item_type = "message"
                content_parts = []
                current_block = TextContent(text="")
                output.content.append(current_block)
                stream.push(TextStartEvent(
                    content_index=block_index(), partial=output
                ))

            elif item.type == "function_call":
                current_item_type = "function_call"
                partial_json = getattr(item, "arguments", "") or ""
                current_block = ToolCall(
                    id=f"{item.call_id}|{item.id}",
                    name=item.name,
                    arguments={},
                )
                output.content.append(current_block)
                stream.push(ToolCallStartEvent(
                    content_index=block_index(), partial=output
                ))

        elif event_type == "response.reasoning_summary_part.added":
            if current_item_type == "reasoning":
                summary_parts.append({"text": getattr(event.part, "text", "")})

        elif event_type == "response.reasoning_summary_text.delta":
            if (
                current_item_type == "reasoning"
                and current_block is not None
                and current_block.type == "thinking"
            ):
                if summary_parts:
                    last_part = summary_parts[-1]
                    current_block.thinking += event.delta
                    last_part["text"] += event.delta
                    stream.push(ThinkingDeltaEvent(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.reasoning_summary_part.done":
            if (
                current_item_type == "reasoning"
                and current_block is not None
                and current_block.type == "thinking"
            ):
                if summary_parts:
                    last_part = summary_parts[-1]
                    current_block.thinking += "\n\n"
                    last_part["text"] += "\n\n"
                    stream.push(ThinkingDeltaEvent(
                        content_index=block_index(),
                        delta="\n\n",
                        partial=output,
                    ))

        elif event_type == "response.content_part.added":
            if current_item_type == "message":
                part = event.part
                part_type = getattr(part, "type", None)
                if part_type in ("output_text", "refusal"):
                    content_parts.append({
                        "type": part_type,
                        "text": getattr(part, "text", ""),
                        "refusal": getattr(part, "refusal", ""),
                    })

        elif event_type == "response.output_text.delta":
            if (
                current_item_type == "message"
                and current_block is not None
                and current_block.type == "text"
            ):
                if not content_parts:
                    continue
                last_part = content_parts[-1]
                if last_part["type"] == "output_text":
                    current_block.text += event.delta
                    last_part["text"] += event.delta
                    stream.push(TextDeltaEvent(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.refusal.delta":
            if (
                current_item_type == "message"
                and current_block is not None
                and current_block.type == "text"
            ):
                if not content_parts:
                    continue
                last_part = content_parts[-1]
                if last_part["type"] == "refusal":
                    current_block.text += event.delta
                    last_part["refusal"] += event.delta
                    stream.push(TextDeltaEvent(
                        content_index=block_index(),
                        delta=event.delta,
                        partial=output,
                    ))

        elif event_type == "response.function_call_arguments.delta":
            if (
                current_item_type == "function_call"
                and current_block is not None
                and current_block.type == "toolCall"
            ):
                partial_json += event.delta
                current_block.arguments = parse_streaming_json(partial_json)
                stream.push(ToolCallDeltaEvent(
                    content_index=block_index(),
                    delta=event.delta,
                    partial=output,
                ))

        elif event_type == "response.function_call_arguments.done":
            if (
                current_item_type == "function_call"
                and current_block is not None
                and current_block.type == "toolCall"
            ):
                previous_partial_json = partial_json
                partial_json = event.arguments
                current_block.arguments = parse_streaming_json(partial_json)

                # Emit trailing delta if final extends previous (TS lines 415-425)
                if event.arguments.startswith(previous_partial_json):
                    delta = event.arguments[len(previous_partial_json):]
                    if delta:
                        stream.push(ToolCallDeltaEvent(
                            content_index=block_index(),
                            delta=delta,
                            partial=output,
                        ))

        elif event_type == "response.output_item.done":
            item = event.item

            if item.type == "reasoning" and current_block is not None and current_block.type == "thinking":
                # Finalize thinking from item summary (TS lines 431-432)
                summary = getattr(item, "summary", None) or []
                current_block.thinking = "\n\n".join(
                    getattr(s, "text", "") for s in summary
                ) if summary else ""
                current_block.thinking_signature = json.dumps(
                    item.model_dump() if hasattr(item, "model_dump")
                    else item.__dict__ if hasattr(item, "__dict__")
                    else {},
                    default=str,
                )
                stream.push(ThinkingEndEvent(
                    content_index=block_index(),
                    content=current_block.thinking,
                    partial=output,
                ))
                current_block = None

            elif item.type == "message" and current_block is not None and current_block.type == "text":
                # Finalize text from item content (TS lines 441-442)
                item_content = getattr(item, "content", []) or []
                current_block.text = "".join(
                    getattr(c, "text", "") if getattr(c, "type", "") == "output_text"
                    else getattr(c, "refusal", "")
                    for c in item_content
                )
                phase = getattr(item, "phase", None)
                current_block.text_signature = encode_text_signature_v1(
                    item.id, phase if phase else None
                )
                stream.push(TextEndEvent(
                    content_index=block_index(),
                    content=current_block.text,
                    partial=output,
                ))
                current_block = None

            elif item.type == "function_call":
                # Finalize tool call (TS lines 450-463)
                args = (
                    parse_streaming_json(partial_json)
                    if partial_json
                    else parse_streaming_json(getattr(item, "arguments", "{}") or "{}")
                )
                tool_call = ToolCall(
                    id=f"{item.call_id}|{item.id}",
                    name=item.name,
                    arguments=args,
                )
                stream.push(ToolCallEndEvent(
                    content_index=block_index(),
                    tool_call=tool_call,
                    partial=output,
                ))
                current_block = None

        elif event_type == "response.completed":
            response = event.response
            if response and getattr(response, "id", None):
                output.response_id = response.id

            usage = getattr(response, "usage", None)
            if usage:
                input_details = getattr(usage, "input_tokens_details", None)
                cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
                output.usage = Usage(
                    input=(getattr(usage, "input_tokens", 0) or 0) - cached_tokens,
                    output=getattr(usage, "output_tokens", 0) or 0,
                    cache_read=cached_tokens,
                    cache_write=0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )

            calculate_cost(model, output.usage)

            if options and options.apply_service_tier_pricing:
                svc_tier = getattr(response, "service_tier", None) or (
                    options.service_tier if options else None
                )
                options.apply_service_tier_pricing(output.usage, svc_tier)

            # Map status to stop reason (TS lines 488-491)
            output.stop_reason = map_stop_reason(getattr(response, "status", None))
            if any(b.type == "toolCall" for b in output.content) and output.stop_reason == "stop":
                output.stop_reason = "toolUse"

        elif event_type == "error":
            code = getattr(event, "code", "unknown")
            message = getattr(event, "message", "Unknown error")
            raise RuntimeError(f"Error Code {code}: {message}")

        elif event_type == "response.failed":
            response = getattr(event, "response", None)
            error = getattr(response, "error", None) if response else None
            details = getattr(response, "incomplete_details", None) if response else None
            if error:
                code = getattr(error, "code", "unknown")
                message = getattr(error, "message", "no message")
                raise RuntimeError(f"{code}: {message}")
            elif details and getattr(details, "reason", None):
                raise RuntimeError(f"incomplete: {details.reason}")
            else:
                raise RuntimeError("Unknown error (no error details in response)")


def _item_to_dict(item: Any) -> dict:
    """Fallback dict conversion for SDK objects without model_dump."""
    if hasattr(item, "__dict__"):
        return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
    return {}
