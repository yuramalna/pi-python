"""Two-pass message normalization for cross-provider compatibility."""

from __future__ import annotations

import time
from collections.abc import Callable

from pi_ai.types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """Normalize messages for cross-provider compatibility (mirrors TS lines 8-172).

    Pass 1: Transform thinking blocks, strip cross-model signatures, normalize IDs.
    Pass 2: Insert synthetic tool results for orphaned tool calls.
    """
    tool_call_id_map: dict[str, str] = {}

    # -- Pass 1: transform each message --
    transformed: list[Message] = []
    for msg in messages:
        if msg.role == "user":
            transformed.append(msg)
            continue

        if msg.role == "toolResult":
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                transformed.append(msg.model_copy(update={"tool_call_id": normalized_id}))
            else:
                transformed.append(msg)
            continue

        if msg.role == "assistant":
            assistant_msg: AssistantMessage = msg  # type: ignore[assignment]
            is_same_model = (
                assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
                and assistant_msg.model == model.id
            )

            new_content = []
            for block in assistant_msg.content:
                if block.type == "thinking":
                    # Redacted thinking: only valid for same model (TS lines 44-46)
                    if block.redacted:
                        if is_same_model:
                            new_content.append(block)
                        continue
                    # Same model + signature: keep for replay (TS line 49)
                    if is_same_model and block.thinking_signature:
                        new_content.append(block)
                        continue
                    # Empty thinking: skip (TS line 51)
                    if not block.thinking or block.thinking.strip() == "":
                        continue
                    # Same model: keep as-is
                    if is_same_model:
                        new_content.append(block)
                        continue
                    # Cross-model: convert to text (TS lines 53-56)
                    new_content.append(TextContent(text=block.thinking))
                    continue

                if block.type == "text":
                    if is_same_model:
                        new_content.append(block)
                    else:
                        # Strip text_signature for cross-model (TS lines 60-65)
                        new_content.append(TextContent(text=block.text))
                    continue

                if block.type == "toolCall":
                    tool_call: ToolCall = block  # type: ignore[assignment]
                    normalized_tool_call = tool_call

                    # Strip thoughtSignature for cross-model (TS lines 71-74)
                    if not is_same_model and tool_call.thought_signature:
                        normalized_tool_call = tool_call.model_copy(
                            update={"thought_signature": None}
                        )

                    # Normalize tool call ID (TS lines 76-82)
                    if not is_same_model and normalize_tool_call_id:
                        normalized_id = normalize_tool_call_id(
                            tool_call.id, model, assistant_msg
                        )
                        if normalized_id != tool_call.id:
                            tool_call_id_map[tool_call.id] = normalized_id
                            normalized_tool_call = normalized_tool_call.model_copy(
                                update={"id": normalized_id}
                            )

                    new_content.append(normalized_tool_call)
                    continue

                # Unknown block type: pass through
                new_content.append(block)

            transformed.append(assistant_msg.model_copy(update={"content": new_content}))
            continue

        # Unknown role: pass through
        transformed.append(msg)

    # -- Pass 2: insert synthetic tool results for orphaned tool calls (TS lines 98-169) --
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def _flush_orphaned() -> None:
        """Insert synthetic error results for orphaned tool calls."""
        for tc in pending_tool_calls:
            if tc.id not in existing_tool_result_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=int(time.time() * 1000),
                    )
                )

    for msg in transformed:
        if msg.role == "assistant":
            # Flush orphaned calls from previous assistant (TS lines 109-124)
            if pending_tool_calls:
                _flush_orphaned()
                pending_tool_calls = []
                existing_tool_result_ids = set()

            # Skip errored/aborted assistant messages (TS lines 130-134)
            assistant_msg = msg  # type: ignore[assignment]
            if assistant_msg.stop_reason in ("error", "aborted"):
                continue

            # Track tool calls from this assistant message (TS lines 137-141)
            tool_calls = [b for b in assistant_msg.content if b.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls  # type: ignore[assignment]
                existing_tool_result_ids = set()

            result.append(msg)

        elif msg.role == "toolResult":
            existing_tool_result_ids.add(msg.tool_call_id)  # type: ignore[union-attr]
            result.append(msg)

        elif msg.role == "user":
            # User interrupts tool flow (TS lines 148-164)
            if pending_tool_calls:
                _flush_orphaned()
                pending_tool_calls = []
                existing_tool_result_ids = set()
            result.append(msg)

        else:
            result.append(msg)

    return result
