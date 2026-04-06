"""Agent loop — the core execution engine.

Implements the multi-turn loop: prompt → LLM → tool calls → repeat.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any

from pi_agent.cancellation import CancellationToken
from pi_agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi_agent.validation import validate_agent_tool_arguments
from pi_ai.stream import stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from pi_ai.utils.event_stream import EventStream

# ---------------------------------------------------------------------------
# Internal types (not exported)
# ---------------------------------------------------------------------------


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass
class _ImmediateToolCallOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass
class _ExecutedToolCallOutcome:
    result: AgentToolResult
    is_error: bool


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> None:
    """Await if coroutine/awaitable, else no-op. Allows sync/async emit callbacks."""
    if inspect.isawaitable(value):
        await value


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _create_agent_stream() -> EventStream[AgentEvent, list[Any]]:
    """Create an EventStream that completes on ``agent_end``."""
    return EventStream(
        is_complete=lambda event: isinstance(event, AgentEndEvent),
        extract_result=lambda event: event.messages if isinstance(event, AgentEndEvent) else [],
    )


def agent_loop(
    prompts: list[Any],
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[Any]]:
    """Start the agent loop and return an event stream.

    Runs in a background asyncio task. Use ``async for event in stream``
    to consume events.

    Args:
        prompts: Initial prompt messages.
        context: Agent context snapshot.
        config: Loop configuration.
        cancellation: Optional cancellation token.
        stream_fn: Optional custom stream function.

    Returns:
        An async iterable ``EventStream`` of ``AgentEvent``.
    """
    stream = _create_agent_stream()

    async def _drive() -> None:
        messages = await run_agent_loop(
            prompts, context, config, lambda e: stream.push(e), cancellation, stream_fn,
        )
        stream.end(messages)

    asyncio.get_running_loop().create_task(_drive())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[Any]]:
    """Continue the agent loop from existing context. Returns an event stream.

    The last message in context must be ``user`` or ``toolResult``.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if hasattr(last, "role") and last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _drive() -> None:
        messages = await run_agent_loop_continue(
            context, config, lambda e: stream.push(e), cancellation, stream_fn,
        )
        stream.end(messages)

    asyncio.get_running_loop().create_task(_drive())
    return stream


async def run_agent_loop(
    prompts: list[Any],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[Any]:
    """Start the agent loop with prompt messages.

    This is the direct async entry point (no background task). Used by
    the ``Agent`` class internally.

    Returns:
        List of all new messages added during the run.
    """
    new_messages: list[Any] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))
    for prompt in prompts:
        await _maybe_await(emit(AgentMessageStartEvent(message=prompt)))
        await _maybe_await(emit(AgentMessageEndEvent(message=prompt)))

    await _run_loop(current_context, new_messages, config, cancellation, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancellation: CancellationToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[Any]:
    """Continue the agent loop from existing context (direct async).

    Returns:
        List of all new messages added during the run.
    """
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if hasattr(last, "role") and last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[Any] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))

    await _run_loop(current_context, new_messages, config, cancellation, emit, stream_fn)
    return new_messages


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[Any],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> None:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    pending_messages: list[Any] = (
        (await config.get_steering_messages()) if config.get_steering_messages else []
    )

    while True:  # Outer loop: follow-ups
        has_more_tool_calls = True

        while has_more_tool_calls or len(pending_messages) > 0:  # Inner loop
            if not first_turn:
                await _maybe_await(emit(TurnStartEvent()))
            else:
                first_turn = False

            # Inject pending messages
            if pending_messages:
                for msg in pending_messages:
                    await _maybe_await(emit(AgentMessageStartEvent(message=msg)))
                    await _maybe_await(emit(AgentMessageEndEvent(message=msg)))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(
                current_context, config, cancellation, emit, stream_fn,
            )
            new_messages.append(message)

            # Error/abort → exit
            if message.stop_reason in ("error", "aborted"):
                await _maybe_await(emit(TurnEndEvent(message=message, tool_results=[])))
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_results = await _execute_tool_calls(
                    current_context, message, config, cancellation, emit,
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _maybe_await(emit(TurnEndEvent(message=message, tool_results=tool_results)))

            # Poll for steering
            pending_messages = (
                (await config.get_steering_messages()) if config.get_steering_messages else []
            )

        # Agent would stop — check follow-ups
        follow_ups: list[Any] = (
            (await config.get_follow_up_messages()) if config.get_follow_up_messages else []
        )
        if follow_ups:
            pending_messages = follow_ups
            continue
        break

    await _maybe_await(emit(AgentEndEvent(messages=new_messages)))


# ---------------------------------------------------------------------------
# Stream assistant response
# ---------------------------------------------------------------------------


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
    stream_fn: StreamFn | None = None,
) -> AssistantMessage:
    """Stream one assistant response from the LLM."""
    # 1. Apply context transform
    messages = context.messages
    if config.transform_context:
        result = config.transform_context(messages, cancellation)
        if inspect.isawaitable(result):
            messages = await result
        else:
            messages = result

    # 2. Convert to LLM messages
    llm_result = config.convert_to_llm(messages)
    if inspect.isawaitable(llm_result):
        llm_messages = await llm_result
    else:
        llm_messages = llm_result

    # 3. Build LLM context
    llm_tools = [t.to_tool() for t in context.tools] if context.tools else []
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=llm_tools,
    )

    # 4. Resolve API key
    resolved_api_key = None
    if config.get_api_key:
        key_result = config.get_api_key(config.model.provider)
        if inspect.isawaitable(key_result):
            resolved_api_key = await key_result
        else:
            resolved_api_key = key_result
    resolved_api_key = resolved_api_key or config.api_key

    # 5. Build stream options and call
    reasoning = config.reasoning if config.reasoning != "off" else None
    options = SimpleStreamOptions(
        reasoning=reasoning,
        thinking_budgets=config.thinking_budgets,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=resolved_api_key,
        cancel_event=cancellation._event if cancellation else None,
        session_id=config.session_id,
        max_retry_delay_ms=config.max_retry_delay_ms,
        on_payload=config.on_payload,
        cache_retention=config.cache_retention,
        headers=config.headers,
        metadata=config.metadata,
    )

    sfn = stream_fn or stream_simple
    response = sfn(config.model, llm_context, options)
    if inspect.isawaitable(response):
        response = await response

    # 6. Process stream events
    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _maybe_await(emit(AgentMessageStartEvent(message=partial_message)))

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _maybe_await(emit(AgentMessageUpdateEvent(
                    message=partial_message,
                    assistant_message_event=event,
                )))

        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await _maybe_await(emit(AgentMessageStartEvent(message=final_message)))
            await _maybe_await(emit(AgentMessageEndEvent(message=final_message)))
            return final_message

    # Fallback if stream ends without done/error
    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _maybe_await(emit(AgentMessageStartEvent(message=final_message)))
    await _maybe_await(emit(AgentMessageEndEvent(message=final_message)))
    return final_message


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Execute tool calls — dispatch to sequential or parallel."""
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]
    if config.tool_execution == "sequential":
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, cancellation, emit,
        )
    return await _execute_tool_calls_parallel(
        current_context, assistant_message, tool_calls, config, cancellation, emit,
    )


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Execute tool calls one at a time."""
    results: list[ToolResultMessage] = []

    for tc in tool_calls:
        await _maybe_await(emit(ToolExecutionStartEvent(
            tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments,
        )))

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tc, config, cancellation,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(
                tc, preparation.result, preparation.is_error, emit,
            ))
        else:
            executed = await _execute_prepared_tool_call(preparation, cancellation, emit)
            results.append(await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed,
                config, cancellation, emit,
            ))

    return results


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Execute tool calls concurrently, emit results in source order."""
    results: list[ToolResultMessage] = []
    runnable_calls: list[_PreparedToolCall] = []

    # Prepare all sequentially (validation, beforeToolCall hook)
    for tc in tool_calls:
        await _maybe_await(emit(ToolExecutionStartEvent(
            tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments,
        )))

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tc, config, cancellation,
        )
        if isinstance(preparation, _ImmediateToolCallOutcome):
            results.append(await _emit_tool_call_outcome(
                tc, preparation.result, preparation.is_error, emit,
            ))
        else:
            runnable_calls.append(preparation)

    # Launch all prepared tools concurrently
    running = [
        {
            "prepared": p,
            "execution": asyncio.ensure_future(
                _execute_prepared_tool_call(p, cancellation, emit),
            ),
        }
        for p in runnable_calls
    ]

    # Await in order (preserves source order)
    for r in running:
        executed = await r["execution"]
        results.append(await _finalize_executed_tool_call(
            current_context, assistant_message, r["prepared"], executed,
            config, cancellation, emit,
        ))

    return results


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: ToolCall) -> ToolCall:
    """Apply tool.prepare_arguments if defined."""
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    return tool_call.model_copy(update={"arguments": prepared_arguments})


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    """Find tool, validate, run beforeToolCall hook."""
    tool = next((t for t in (current_context.tools or []) if t.name == tool_call.name), None)
    if not tool:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_agent_tool_arguments(tool, prepared_tool_call)

        if config.before_tool_call:
            before_result = await config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=validated_args,
                    context=current_context,
                ),
                cancellation,
            )
            if before_result and before_result.block:
                return _ImmediateToolCallOutcome(
                    result=_create_error_tool_result(
                        before_result.reason or "Tool execution was blocked",
                    ),
                    is_error=True,
                )

        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as e:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(str(e)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    """Call tool.execute with update callback."""
    update_tasks: list[Any] = []

    def on_update(partial_result: AgentToolResult) -> None:
        result = emit(ToolExecutionUpdateEvent(
            tool_call_id=prepared.tool_call.id,
            tool_name=prepared.tool_call.name,
            args=prepared.tool_call.arguments,
            partial_result=partial_result,
        ))
        if inspect.isawaitable(result):
            update_tasks.append(result)

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            cancellation,
            on_update,
        )
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as e:
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return _ExecutedToolCallOutcome(
            result=_create_error_tool_result(str(e)),
            is_error=True,
        )


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    cancellation: CancellationToken | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Run afterToolCall hook and emit outcome."""
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call:
        after_result = await config.after_tool_call(
            AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared.tool_call,
                args=prepared.args,
                result=result,
                is_error=is_error,
                context=current_context,
            ),
            cancellation,
        )
        if after_result:
            result = AgentToolResult(
                content=after_result.content if after_result.content is not None else result.content,
                details=after_result.details if after_result.details is not None else result.details,
            )
            is_error = after_result.is_error if after_result.is_error is not None else is_error

    return await _emit_tool_call_outcome(prepared.tool_call, result, is_error, emit)


def _create_error_tool_result(message: str) -> AgentToolResult:
    """Create an error tool result with a text message."""
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _emit_tool_call_outcome(
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Emit tool_execution_end + message_start/end events."""
    await _maybe_await(emit(ToolExecutionEndEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=result,
        is_error=is_error,
    )))

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
        timestamp=int(time.time() * 1000),
    )

    await _maybe_await(emit(AgentMessageStartEvent(message=tool_result_message)))
    await _maybe_await(emit(AgentMessageEndEvent(message=tool_result_message)))
    return tool_result_message
