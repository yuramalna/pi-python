"""Tests for the agent loop.

Port of reference/packages/agent/test/agent-loop.test.ts.
"""

import asyncio
import time

import pytest

from pi_llm_agent import (
    AgentContext,
    AgentLoopConfig,
    AgentMessageEndEvent,
    AgentTool,
    AgentToolResult,
    agent_loop,
    agent_loop_continue,
)
from pi_llm_agent.types import (
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi_llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_model() -> Model:
    return Model(id="mock", name="mock", api="openai-responses", provider="openai")


def _create_assistant_message(
    content: list, stop_reason: str = "stop",
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def _create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=int(time.time() * 1000))


def _identity_converter(messages):
    """Pass through standard message roles, filter out custom ones."""
    return [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")]


def _make_stream_fn(response_message: AssistantMessage):
    """Create a stream function that returns a canned response."""
    def stream_fn(*_args, **_kwargs):
        s = AssistantMessageEventStream()

        async def _push():
            await asyncio.sleep(0)
            s.push(DoneEvent(reason=response_message.stop_reason, message=response_message))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    return stream_fn


# ---------------------------------------------------------------------------
# agentLoop tests
# ---------------------------------------------------------------------------


async def test_emit_events_with_agent_message_types():
    """Basic prompt → text response, verify event sequence. (TS test 1)"""
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
    user_prompt = _create_user_message("Hello")
    config = AgentLoopConfig(model=_create_model(), convert_to_llm=_identity_converter)

    response_msg = _create_assistant_message([TextContent(text="Hi there!")])
    stream = agent_loop([user_prompt], context, config, stream_fn=_make_stream_fn(response_msg))

    events = []
    async for event in stream:
        events.append(event)

    messages = await stream.result()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    event_types = [e.type for e in events]
    assert "agent_start" in event_types
    assert "turn_start" in event_types
    assert "message_start" in event_types
    assert "message_end" in event_types
    assert "turn_end" in event_types
    assert "agent_end" in event_types


async def test_handle_custom_message_types_via_convert_to_llm():
    """Custom message types are filtered by convertToLlm. (TS test 2)"""

    class _Notification:
        role = "notification"
        text = "This is a notification"
        timestamp = int(time.time() * 1000)

    notification = _Notification()
    context = AgentContext(system_prompt="You are helpful.", messages=[notification], tools=[])
    user_prompt = _create_user_message("Hello")

    converted_messages = []

    def converter(messages):
        result = [
            m for m in messages
            if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
        ]
        converted_messages.extend(result)
        return result

    config = AgentLoopConfig(model=_create_model(), convert_to_llm=converter)
    response_msg = _create_assistant_message([TextContent(text="Response")])
    stream = agent_loop([user_prompt], context, config, stream_fn=_make_stream_fn(response_msg))

    async for _ in stream:
        pass

    # Notification should have been filtered out — only user message passes
    assert len(converted_messages) == 1
    assert converted_messages[0].role == "user"


async def test_apply_transform_context_before_convert_to_llm():
    """transformContext prunes messages before convertToLlm. (TS test 3)"""
    context = AgentContext(
        system_prompt="You are helpful.",
        messages=[
            _create_user_message("old message 1"),
            _create_assistant_message([TextContent(text="old response 1")]),
            _create_user_message("old message 2"),
            _create_assistant_message([TextContent(text="old response 2")]),
        ],
        tools=[],
    )
    user_prompt = _create_user_message("new message")

    transformed_messages = []
    converted_messages = []

    async def transform_context(messages, _cancellation=None):
        result = messages[-2:]
        transformed_messages.extend(result)
        return result

    def converter(messages):
        result = _identity_converter(messages)
        converted_messages.extend(result)
        return result

    config = AgentLoopConfig(
        model=_create_model(),
        convert_to_llm=converter,
        transform_context=transform_context,
    )
    response_msg = _create_assistant_message([TextContent(text="Response")])
    stream = agent_loop([user_prompt], context, config, stream_fn=_make_stream_fn(response_msg))

    async for _ in stream:
        pass

    assert len(transformed_messages) == 2
    assert len(converted_messages) == 2


async def test_handle_tool_calls_and_results():
    """Tool call round-trip: prompt → tool call → tool result → final. (TS test 4)"""
    executed = []

    class EchoTool(AgentTool):
        async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
            executed.append(params["value"])
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params['value']}")],
                details={"value": params["value"]},
            )

    tool = EchoTool(
        name="echo", label="Echo", description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    user_prompt = _create_user_message("echo something")
    config = AgentLoopConfig(model=_create_model(), convert_to_llm=_identity_converter)

    call_index = 0

    def stream_fn(*_args, **_kwargs):
        nonlocal call_index
        s = AssistantMessageEventStream()

        idx = call_index
        call_index += 1

        async def _push():
            await asyncio.sleep(0)
            if idx == 0:
                msg = _create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})],
                    stop_reason="toolUse",
                )
                s.push(DoneEvent(reason="toolUse", message=msg))
            else:
                msg = _create_assistant_message([TextContent(text="done")])
                s.push(DoneEvent(reason="stop", message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    events = []
    stream = agent_loop([user_prompt], context, config, stream_fn=stream_fn)
    async for event in stream:
        events.append(event)

    assert executed == ["hello"]

    tool_start = next((e for e in events if isinstance(e, ToolExecutionStartEvent)), None)
    tool_end = next((e for e in events if isinstance(e, ToolExecutionEndEvent)), None)
    assert tool_start is not None
    assert tool_end is not None
    assert tool_end.is_error is False


async def test_execute_mutated_before_tool_call_args():
    """beforeToolCall can mutate args in place. (TS test 5)"""
    executed = []

    class EchoTool(AgentTool):
        async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
            executed.append(params["value"])
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params['value']}")],
                details={"value": params["value"]},
            )

    tool = EchoTool(
        name="echo", label="Echo", description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    user_prompt = _create_user_message("echo something")

    async def before_tool_call(ctx, _cancellation=None):
        ctx.args["value"] = 123
        return None

    config = AgentLoopConfig(
        model=_create_model(),
        convert_to_llm=_identity_converter,
        before_tool_call=before_tool_call,
    )

    call_index = 0

    def stream_fn(*_args, **_kwargs):
        nonlocal call_index
        s = AssistantMessageEventStream()
        idx = call_index
        call_index += 1

        async def _push():
            await asyncio.sleep(0)
            if idx == 0:
                msg = _create_assistant_message(
                    [ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})],
                    stop_reason="toolUse",
                )
                s.push(DoneEvent(reason="toolUse", message=msg))
            else:
                msg = _create_assistant_message([TextContent(text="done")])
                s.push(DoneEvent(reason="stop", message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    stream = agent_loop([user_prompt], context, config, stream_fn=stream_fn)
    async for _ in stream:
        pass

    assert executed == [123]


async def test_prepare_tool_arguments_for_validation():
    """prepareArguments reshapes flat args into nested structure. (TS test 6)"""
    executed = []

    class EditTool(AgentTool):
        def prepare_arguments(self, args):
            if not isinstance(args, dict):
                return args
            if "oldText" in args and "newText" in args:
                return {
                    "edits": [
                        *args.get("edits", []),
                        {"oldText": args["oldText"], "newText": args["newText"]},
                    ],
                }
            return args

        async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
            executed.append(params["edits"])
            return AgentToolResult(
                content=[TextContent(text=f"edited {len(params['edits'])}")],
                details={"count": len(params["edits"])},
            )

    tool = EditTool(
        name="edit", label="Edit", description="Edit tool",
        parameters={
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["edits"],
        },
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    user_prompt = _create_user_message("edit something")
    config = AgentLoopConfig(model=_create_model(), convert_to_llm=_identity_converter)

    call_index = 0

    def stream_fn(*_args, **_kwargs):
        nonlocal call_index
        s = AssistantMessageEventStream()
        idx = call_index
        call_index += 1

        async def _push():
            await asyncio.sleep(0)
            if idx == 0:
                msg = _create_assistant_message(
                    [ToolCall(
                        id="tool-1", name="edit",
                        arguments={"oldText": "before", "newText": "after"},
                    )],
                    stop_reason="toolUse",
                )
                s.push(DoneEvent(reason="toolUse", message=msg))
            else:
                msg = _create_assistant_message([TextContent(text="done")])
                s.push(DoneEvent(reason="stop", message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    stream = agent_loop([user_prompt], context, config, stream_fn=stream_fn)
    async for _ in stream:
        pass

    assert executed == [[{"oldText": "before", "newText": "after"}]]


async def test_execute_tool_calls_parallel_source_order():
    """Parallel execution runs concurrently, results emitted in source order. (TS test 7)"""
    first_resolved = False
    parallel_observed = False
    release_first: asyncio.Event = asyncio.Event()

    class EchoTool(AgentTool):
        async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
            nonlocal first_resolved, parallel_observed
            if params["value"] == "first":
                await asyncio.wait_for(release_first.wait(), timeout=2.0)
                first_resolved = True
            if params["value"] == "second" and not first_resolved:
                parallel_observed = True
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params['value']}")],
                details={"value": params["value"]},
            )

    tool = EchoTool(
        name="echo", label="Echo", description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    user_prompt = _create_user_message("echo both")
    config = AgentLoopConfig(
        model=_create_model(),
        convert_to_llm=_identity_converter,
        tool_execution="parallel",
    )

    call_index = 0

    def stream_fn(*_args, **_kwargs):
        nonlocal call_index
        s = AssistantMessageEventStream()
        idx = call_index
        call_index += 1

        async def _push():
            await asyncio.sleep(0)
            if idx == 0:
                msg = _create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    stop_reason="toolUse",
                )
                s.push(DoneEvent(reason="toolUse", message=msg))

                async def _release():
                    await asyncio.sleep(0.02)
                    release_first.set()
                asyncio.get_running_loop().create_task(_release())
            else:
                msg = _create_assistant_message([TextContent(text="done")])
                s.push(DoneEvent(reason="stop", message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    events = []
    stream = agent_loop([user_prompt], context, config, stream_fn=stream_fn)
    async for event in stream:
        events.append(event)

    # Verify parallel execution happened
    assert parallel_observed is True

    # Verify results emitted in source order
    tool_result_ids = [
        e.message.tool_call_id
        for e in events
        if isinstance(e, AgentMessageEndEvent)
        and hasattr(e.message, "role") and e.message.role == "toolResult"
    ]
    assert tool_result_ids == ["tool-1", "tool-2"]


async def test_inject_queued_messages_after_tool_calls():
    """Steering messages appear after all tool results complete. (TS test 8)"""
    executed = []

    class EchoTool(AgentTool):
        async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
            executed.append(params["value"])
            return AgentToolResult(
                content=[TextContent(text=f"ok:{params['value']}")],
                details={"value": params["value"]},
            )

    tool = EchoTool(
        name="echo", label="Echo", description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    context = AgentContext(system_prompt="", messages=[], tools=[tool])
    user_prompt = _create_user_message("start")
    queued_user_message = _create_user_message("interrupt")

    queued_delivered = False
    saw_interrupt_in_context = False
    call_index = 0

    async def get_steering_messages():
        nonlocal queued_delivered
        if len(executed) >= 1 and not queued_delivered:
            queued_delivered = True
            return [queued_user_message]
        return []

    config = AgentLoopConfig(
        model=_create_model(),
        convert_to_llm=_identity_converter,
        tool_execution="sequential",
        get_steering_messages=get_steering_messages,
    )

    def stream_fn(_model, ctx, *_args, **_kwargs):
        nonlocal call_index, saw_interrupt_in_context
        s = AssistantMessageEventStream()
        idx = call_index

        if idx == 1:
            saw_interrupt_in_context = any(
                hasattr(m, "role") and m.role == "user"
                and hasattr(m, "content") and m.content == "interrupt"
                for m in ctx.messages
            )

        call_index += 1

        async def _push():
            await asyncio.sleep(0)
            if idx == 0:
                msg = _create_assistant_message(
                    [
                        ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
                        ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
                    ],
                    stop_reason="toolUse",
                )
                s.push(DoneEvent(reason="toolUse", message=msg))
            else:
                msg = _create_assistant_message([TextContent(text="done")])
                s.push(DoneEvent(reason="stop", message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    events = []
    stream = agent_loop([user_prompt], context, config, stream_fn=stream_fn)
    async for event in stream:
        events.append(event)

    # Both tools should execute before steering is injected
    assert executed == ["first", "second"]

    tool_ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    # Queued message should appear in events after both tool result messages
    from pi_llm_agent.types import AgentMessageStartEvent
    event_sequence = []
    for e in events:
        if not isinstance(e, AgentMessageStartEvent):
            continue
        msg = e.message
        if hasattr(msg, "role") and msg.role == "toolResult":
            event_sequence.append(f"tool:{msg.tool_call_id}")
        elif hasattr(msg, "role") and msg.role == "user" and hasattr(msg, "content") and msg.content == "interrupt":
            event_sequence.append("interrupt")

    assert "interrupt" in event_sequence
    assert event_sequence.index("tool:tool-1") < event_sequence.index("interrupt")
    assert event_sequence.index("tool:tool-2") < event_sequence.index("interrupt")

    # Interrupt message should be in context when second LLM call is made
    assert saw_interrupt_in_context is True


# ---------------------------------------------------------------------------
# agentLoopContinue tests
# ---------------------------------------------------------------------------


def test_continue_throws_no_messages():
    """agentLoopContinue raises when context has no messages. (TS test 9)"""
    context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[])
    config = AgentLoopConfig(model=_create_model(), convert_to_llm=_identity_converter)
    with pytest.raises(ValueError, match="Cannot continue: no messages in context"):
        agent_loop_continue(context, config)


async def test_continue_from_existing_context():
    """Continue returns only new assistant message, no user message events. (TS test 10)"""
    user_message = _create_user_message("Hello")
    context = AgentContext(
        system_prompt="You are helpful.",
        messages=[user_message],
        tools=[],
    )
    config = AgentLoopConfig(model=_create_model(), convert_to_llm=_identity_converter)

    response_msg = _create_assistant_message([TextContent(text="Response")])
    stream = agent_loop_continue(context, config, stream_fn=_make_stream_fn(response_msg))

    events = []
    async for event in stream:
        events.append(event)

    messages = await stream.result()

    # Should only return the new assistant message
    assert len(messages) == 1
    assert messages[0].role == "assistant"

    # Should NOT have user message events
    message_end_events = [e for e in events if isinstance(e, AgentMessageEndEvent)]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"


async def test_continue_allows_custom_last_message():
    """Custom role as last message works if convertToLlm handles it. (TS test 11)"""

    class _CustomMessage:
        role = "custom"
        text = "Hook content"
        timestamp = int(time.time() * 1000)

    custom_message = _CustomMessage()
    context = AgentContext(
        system_prompt="You are helpful.",
        messages=[custom_message],
        tools=[],
    )

    def converter(messages):
        result = []
        for m in messages:
            if hasattr(m, "role") and m.role == "custom":
                result.append(UserMessage(content=m.text, timestamp=m.timestamp))
            elif hasattr(m, "role") and m.role in ("user", "assistant", "toolResult"):
                result.append(m)
        return result

    config = AgentLoopConfig(model=_create_model(), convert_to_llm=converter)
    response_msg = _create_assistant_message([TextContent(text="Response to custom message")])
    stream = agent_loop_continue(context, config, stream_fn=_make_stream_fn(response_msg))

    events = []
    async for event in stream:
        events.append(event)

    messages = await stream.result()
    assert len(messages) == 1
    assert messages[0].role == "assistant"
