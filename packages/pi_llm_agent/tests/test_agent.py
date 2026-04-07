"""Tests for the Agent class.

Port of reference/packages/agent/test/agent.test.ts (463 lines, 15 tests).
"""

import asyncio
import time

import pytest

from pi_llm_agent import (
    AgentTool,
)
from pi_llm_agent.agent import (
    Agent,
    AgentOptions,
    InitialAgentState,
    PendingMessageQueue,
)
from pi_llm.models import get_model
from pi_llm.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    Usage,
    UserMessage,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_assistant_message(text: str = "ok") -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def _make_done_stream_fn(message: AssistantMessage | None = None):
    """Stream function that immediately pushes a DoneEvent."""
    msg = message or _create_assistant_message()

    def stream_fn(*_args, **_kwargs):
        s = AssistantMessageEventStream()

        async def _push():
            await asyncio.sleep(0)
            s.push(DoneEvent(reason=msg.stop_reason, message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    return stream_fn


def _make_abort_aware_stream_fn():
    """Stream function that waits for cancellation, then pushes ErrorEvent."""

    def stream_fn(_model, _context, options):
        s = AssistantMessageEventStream()
        cancel_event = options.cancel_event if options else None

        async def _drive():
            await asyncio.sleep(0)
            s.push(StartEvent(partial=_create_assistant_message("")))
            if cancel_event:
                await cancel_event.wait()
            aborted_msg = AssistantMessage(
                content=[TextContent(text="")],
                api="openai-responses",
                provider="openai",
                model="mock",
                usage=Usage(),
                stop_reason="aborted",
                error_message="Aborted",
                timestamp=int(time.time() * 1000),
            )
            s.push(ErrorEvent(reason="aborted", error=aborted_msg))
            s.end()

        asyncio.get_running_loop().create_task(_drive())
        return s

    return stream_fn


# ---------------------------------------------------------------------------
# Test 1: Default state initialization
# ---------------------------------------------------------------------------


def test_default_state():
    agent = Agent()
    assert agent.state.system_prompt == ""
    assert agent.state.model is not None
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == set()
    assert agent.state.error_message is None


# ---------------------------------------------------------------------------
# Test 2: Custom initial state
# ---------------------------------------------------------------------------


def test_custom_initial_state():
    custom_model = get_model("openai", "gpt-4o-mini")
    agent = Agent(
        AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=custom_model,
                thinking_level="low",
            ),
        )
    )
    assert agent.state.system_prompt == "You are a helpful assistant."
    assert agent.state.model is custom_model
    assert agent.state.thinking_level == "low"


# ---------------------------------------------------------------------------
# Test 3: Subscribe to events
# ---------------------------------------------------------------------------


def test_subscribe_to_events():
    agent = Agent()

    event_count = 0

    def listener(_event, _cancellation):
        nonlocal event_count
        event_count += 1

    unsub = agent.subscribe(listener)

    # No initial event on subscribe
    assert event_count == 0

    # State mutators don't emit events
    agent.state.system_prompt = "Test prompt"
    assert event_count == 0
    assert agent.state.system_prompt == "Test prompt"

    # Unsubscribe should work
    unsub()
    agent.state.system_prompt = "Another prompt"
    assert event_count == 0


# ---------------------------------------------------------------------------
# Test 4: Await async subscribers before prompt resolves
# ---------------------------------------------------------------------------


async def test_await_async_subscribers():
    barrier = asyncio.Event()
    agent = Agent(AgentOptions(stream_fn=_make_done_stream_fn()))

    listener_finished = False

    async def listener(event, _cancellation):
        nonlocal listener_finished
        if event.type == "agent_end":
            await barrier.wait()
            listener_finished = True

    agent.subscribe(listener)

    prompt_resolved = False

    async def run_prompt():
        nonlocal prompt_resolved
        await agent.prompt("hello")
        prompt_resolved = True

    task = asyncio.create_task(run_prompt())
    await asyncio.sleep(0.01)

    assert prompt_resolved is False
    assert listener_finished is False
    assert agent.state.is_streaming is True

    barrier.set()
    await task

    assert listener_finished is True
    assert prompt_resolved is True
    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# Test 5: waitForIdle waits for async subscribers
# ---------------------------------------------------------------------------


async def test_wait_for_idle_waits_for_subscribers():
    barrier = asyncio.Event()
    agent = Agent(AgentOptions(stream_fn=_make_done_stream_fn()))

    async def listener(event, _cancellation):
        if (
            event.type == "message_end"
            and hasattr(event.message, "role")
            and event.message.role == "assistant"
        ):
            await barrier.wait()

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))

    idle_resolved = False

    async def wait_idle():
        nonlocal idle_resolved
        await agent.wait_for_idle()
        idle_resolved = True

    idle_task = asyncio.create_task(wait_idle())

    await asyncio.sleep(0.01)
    assert idle_resolved is False
    assert agent.state.is_streaming is True

    barrier.set()
    await asyncio.gather(prompt_task, idle_task)

    assert idle_resolved is True
    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# Test 6: Pass active cancellation to subscribers
# ---------------------------------------------------------------------------


async def test_pass_cancellation_to_subscribers():
    received_cancellation = None
    agent = Agent(AgentOptions(stream_fn=_make_abort_aware_stream_fn()))

    def listener(event, cancellation):
        nonlocal received_cancellation
        if event.type == "agent_start":
            received_cancellation = cancellation

    agent.subscribe(listener)

    prompt_task = asyncio.create_task(agent.prompt("hello"))
    await asyncio.sleep(0.01)

    assert received_cancellation is not None
    assert received_cancellation.is_cancelled is False

    agent.abort()
    await prompt_task

    assert received_cancellation.is_cancelled is True


# ---------------------------------------------------------------------------
# Test 7: Update state with mutators
# ---------------------------------------------------------------------------


def test_state_mutators():
    agent = Agent()

    # systemPrompt
    agent.state.system_prompt = "Custom prompt"
    assert agent.state.system_prompt == "Custom prompt"

    # model
    new_model = get_model("openai", "gpt-4o-mini")
    agent.state.model = new_model
    assert agent.state.model is new_model

    # thinkingLevel
    agent.state.thinking_level = "high"
    assert agent.state.thinking_level == "high"

    # tools — setter copies
    tools = [
        AgentTool(name="test", label="test", description="test tool", parameters={})
    ]
    agent.state.tools = tools
    assert len(agent.state.tools) == 1
    assert agent.state.tools is not tools  # copy

    # messages — setter copies
    messages = [
        UserMessage(content="Hello", timestamp=int(time.time() * 1000))
    ]
    agent.state.messages = messages
    assert len(agent.state.messages) == 1
    assert agent.state.messages is not messages  # copy

    # append to messages (getter returns internal reference)
    new_message = _create_assistant_message("Hi")
    agent.state.messages.append(new_message)
    assert len(agent.state.messages) == 2
    assert agent.state.messages[1] is new_message

    # clear messages
    agent.state.messages = []
    assert agent.state.messages == []


# ---------------------------------------------------------------------------
# Test 8: Steering message queue
# ---------------------------------------------------------------------------


def test_steering_queue():
    agent = Agent()
    message = UserMessage(
        content="Steering message", timestamp=int(time.time() * 1000)
    )
    agent.steer(message)
    assert message not in agent.state.messages


# ---------------------------------------------------------------------------
# Test 9: Follow-up message queue
# ---------------------------------------------------------------------------


def test_follow_up_queue():
    agent = Agent()
    message = UserMessage(
        content="Follow-up message", timestamp=int(time.time() * 1000)
    )
    agent.follow_up(message)
    assert message not in agent.state.messages


# ---------------------------------------------------------------------------
# Test 10: Handle abort controller — no throw when nothing running
# ---------------------------------------------------------------------------


def test_abort_no_throw():
    agent = Agent()
    agent.abort()  # should not raise


# ---------------------------------------------------------------------------
# Test 11: Throw when prompt() called while streaming
# ---------------------------------------------------------------------------


async def test_prompt_throws_when_streaming():
    agent = Agent(AgentOptions(stream_fn=_make_abort_aware_stream_fn()))

    first_task = asyncio.create_task(agent.prompt("First message"))
    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError, match="already processing a prompt"):
        await agent.prompt("Second message")

    agent.abort()
    await first_task


# ---------------------------------------------------------------------------
# Test 12: Throw when continue_() called while streaming
# ---------------------------------------------------------------------------


async def test_continue_throws_when_streaming():
    agent = Agent(AgentOptions(stream_fn=_make_abort_aware_stream_fn()))

    first_task = asyncio.create_task(agent.prompt("First message"))
    await asyncio.sleep(0.01)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()

    agent.abort()
    await first_task


# ---------------------------------------------------------------------------
# Test 13: continue_() processes queued follow-up messages
# ---------------------------------------------------------------------------


async def test_continue_processes_follow_ups():
    agent = Agent(AgentOptions(stream_fn=_make_done_stream_fn()))

    agent.state.messages = [
        UserMessage(
            content=[TextContent(text="Initial")],
            timestamp=int(time.time() * 1000) - 10,
        ),
        _create_assistant_message("Initial response"),
    ]

    agent.follow_up(
        UserMessage(
            content=[TextContent(text="Queued follow-up")],
            timestamp=int(time.time() * 1000),
        )
    )

    await agent.continue_()

    has_queued = any(
        hasattr(m, "role")
        and m.role == "user"
        and _message_contains_text(m, "Queued follow-up")
        for m in agent.state.messages
    )
    assert has_queued is True
    assert agent.state.messages[-1].role == "assistant"


def _message_contains_text(msg: object, text: str) -> bool:
    """Check if a message contains specific text in its content."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return text in content
    if isinstance(content, list):
        return any(
            hasattr(part, "text") and text in part.text
            for part in content
        )
    return False


# ---------------------------------------------------------------------------
# Test 14: continue_() keeps one-at-a-time steering semantics
# ---------------------------------------------------------------------------


async def test_continue_steering_one_at_a_time():
    response_count = 0

    def counting_stream_fn(*_args, **_kwargs):
        nonlocal response_count
        response_count += 1
        count = response_count

        s = AssistantMessageEventStream()

        async def _push():
            await asyncio.sleep(0)
            s.push(
                DoneEvent(
                    reason="stop",
                    message=_create_assistant_message(f"Processed {count}"),
                )
            )
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    agent = Agent(AgentOptions(stream_fn=counting_stream_fn))

    agent.state.messages = [
        UserMessage(
            content=[TextContent(text="Initial")],
            timestamp=int(time.time() * 1000) - 10,
        ),
        _create_assistant_message("Initial response"),
    ]

    agent.steer(
        UserMessage(
            content=[TextContent(text="Steering 1")],
            timestamp=int(time.time() * 1000),
        )
    )
    agent.steer(
        UserMessage(
            content=[TextContent(text="Steering 2")],
            timestamp=int(time.time() * 1000) + 1,
        )
    )

    await agent.continue_()

    recent = agent.state.messages[-4:]
    roles = [m.role for m in recent]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert response_count == 2


# ---------------------------------------------------------------------------
# Test 15: Forwards sessionId to streamFn options
# ---------------------------------------------------------------------------


async def test_forwards_session_id():
    received_session_id = None

    def stream_fn(_model, _context, options):
        nonlocal received_session_id
        received_session_id = options.session_id if options else None

        s = AssistantMessageEventStream()

        async def _push():
            await asyncio.sleep(0)
            s.push(
                DoneEvent(
                    reason="stop", message=_create_assistant_message("ok")
                )
            )
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    agent = Agent(AgentOptions(session_id="session-abc", stream_fn=stream_fn))
    await agent.prompt("hello")
    assert received_session_id == "session-abc"

    # Setter updates it
    agent.session_id = "session-def"
    assert agent.session_id == "session-def"

    await agent.prompt("hello again")
    assert received_session_id == "session-def"


# ---------------------------------------------------------------------------
# PendingMessageQueue unit tests
# ---------------------------------------------------------------------------


def test_pending_message_queue_all_mode():
    q = PendingMessageQueue(mode="all")
    q.enqueue("a")
    q.enqueue("b")
    assert q.has_items() is True
    assert q.drain() == ["a", "b"]
    assert q.drain() == []
    assert q.has_items() is False


def test_pending_message_queue_one_at_a_time_mode():
    q = PendingMessageQueue(mode="one-at-a-time")
    q.enqueue("a")
    q.enqueue("b")
    assert q.drain() == ["a"]
    assert q.drain() == ["b"]
    assert q.drain() == []


def test_pending_message_queue_clear():
    q = PendingMessageQueue(mode="all")
    q.enqueue("a")
    q.enqueue("b")
    q.clear()
    assert q.has_items() is False
    assert q.drain() == []
