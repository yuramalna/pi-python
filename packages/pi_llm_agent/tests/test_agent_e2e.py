"""End-to-end agent tests using the faux provider.

Ported from reference/packages/agent/test/e2e.test.ts.
These tests exercise the full agent pipeline (Agent → agent_loop → stream)
with deterministic scripted responses.
"""

import asyncio
import time

import pytest

from pi_llm_agent import (
    Agent,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    InitialAgentState,
)
from pi_llm import TextContent, ThinkingContent, ToolCall, UserMessage, stream_simple
from pi_llm.providers.faux import (
    FauxModelDefinition,
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
    register_faux_provider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_contains(msg, needle: str) -> bool:
    """Check if a message contains a substring, handling both string and list content."""
    if not hasattr(msg, "role") or msg.role != "user":
        return False
    if isinstance(msg.content, str):
        return needle in msg.content
    if isinstance(msg.content, list):
        return any(
            hasattr(b, "text") and needle in b.text
            for b in msg.content
        )
    return False


_registrations: list = []


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for reg in _registrations:
        reg.unregister()
    _registrations.clear()


def _faux(**kw):
    reg = register_faux_provider(RegisterFauxProviderOptions(**kw) if kw else None)
    _registrations.append(reg)
    return reg


def _now():
    return int(time.time() * 1000)


class CalculateTool(AgentTool):
    """Simple calculator tool for testing."""

    def __init__(self):
        super().__init__(
            name="calculate",
            label="Calculator",
            description="Evaluate a math expression",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        expr = params.get("expression", "")
        try:
            result = eval(expr)  # noqa: S307 — safe for testing
            return AgentToolResult(content=[TextContent(text=f"{expr} = {result}")])
        except Exception as e:
            return AgentToolResult(content=[TextContent(text=f"Error: {e}")])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicPrompt:
    @pytest.mark.asyncio
    async def test_handles_basic_text_prompt(self):
        faux = _faux()
        faux.set_responses([faux_assistant_message("4")])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="off",
                tools=[],
            ),
            stream_fn=stream_simple,
        ))

        await agent.prompt("What is 2+2?")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[0].role == "user"
        assert agent.state.messages[1].role == "assistant"
        assert agent.state.messages[1].content[0].text == "4"


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_executes_tools_and_tracks_pending(self):
        faux = _faux()
        faux.set_responses([
            faux_assistant_message(
                [
                    faux_text("Let me calculate that."),
                    faux_tool_call("calculate", {"expression": "123 * 456"}, id="calc-1"),
                ],
                stop_reason="toolUse",
            ),
            faux_assistant_message("The result is 56088."),
        ])

        pending_during_events = []
        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="Use calculator for math.",
                model=faux.get_model(),
                thinking_level="off",
                tools=[CalculateTool()],
            ),
            stream_fn=stream_simple,
        ))

        def on_event(event, cancellation):
            if event.type in ("tool_execution_start", "tool_execution_end"):
                pending_during_events.append({
                    "type": event.type,
                    "ids": set(agent.state.pending_tool_calls),
                })

        agent.subscribe(on_event)
        await agent.prompt("Calculate 123 * 456")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) >= 4

        # Find tool result
        tool_results = [m for m in agent.state.messages if hasattr(m, "role") and m.role == "toolResult"]
        assert len(tool_results) == 1
        assert "56088" in tool_results[0].content[0].text

        # Final message has the answer
        last = agent.state.messages[-1]
        assert last.role == "assistant"
        assert "56088" in last.content[0].text

        # Pending tool calls tracked correctly
        assert len(agent.state.pending_tool_calls) == 0
        assert len(pending_during_events) == 2
        assert pending_during_events[0]["type"] == "tool_execution_start"
        assert "calc-1" in pending_during_events[0]["ids"]
        assert pending_during_events[1]["type"] == "tool_execution_end"
        assert len(pending_during_events[1]["ids"]) == 0


class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_during_streaming(self):
        faux = _faux(tokens_per_second=20, token_size_min=2, token_size_max=2)
        faux.set_responses([
            faux_assistant_message(
                "one two three four five six seven eight nine ten eleven twelve"
            ),
        ])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="off",
            ),
            stream_fn=stream_simple,
        ))

        prompt_task = asyncio.create_task(agent.prompt("Count slowly."))
        await asyncio.sleep(0.05)
        agent.abort()
        await prompt_task

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) >= 2
        last = agent.state.messages[-1]
        assert last.role == "assistant"
        assert last.stop_reason == "aborted"
        assert agent.state.error_message is not None


class TestEventLifecycle:
    @pytest.mark.asyncio
    async def test_emits_lifecycle_events(self):
        faux = _faux(token_size_min=1, token_size_max=1)
        faux.set_responses([faux_assistant_message("1 2 3 4 5")])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="off",
            ),
            stream_fn=stream_simple,
        ))

        events = []
        agent.subscribe(lambda event, cancel: events.append(event.type))
        await agent.prompt("Count from 1 to 5.")

        assert "agent_start" in events
        assert "turn_start" in events
        assert "message_start" in events
        assert "message_update" in events
        assert "message_end" in events
        assert "turn_end" in events
        assert "agent_end" in events
        # Ordering
        assert events.index("agent_start") < events.index("message_start")
        assert events.index("message_start") < events.index("message_end")
        assert events.index("message_end") < events.index("agent_end")

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_maintains_context_across_turns(self):
        faux = _faux()
        faux.set_responses([
            faux_assistant_message("Nice to meet you, Alice."),
            lambda ctx, opts, state, model: faux_assistant_message(
                "Your name is Alice."
                if any(
                    _message_contains(m, "Alice")
                    for m in ctx.messages
                )
                else "I do not know your name."
            ),
        ])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="off",
            ),
            stream_fn=stream_simple,
        ))

        await agent.prompt("My name is Alice.")
        assert len(agent.state.messages) == 2

        await agent.prompt("What is my name?")
        assert len(agent.state.messages) == 4

        last = agent.state.messages[3]
        assert "Alice" in last.content[0].text


class TestThinkingContent:
    @pytest.mark.asyncio
    async def test_preserves_thinking_blocks(self):
        faux = _faux(models=[FauxModelDefinition(id="faux-reasoning", reasoning=True)])
        faux.set_responses([
            faux_assistant_message([faux_thinking("step by step"), faux_text("4")]),
        ])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="low",
            ),
            stream_fn=stream_simple,
        ))

        await agent.prompt("What is 2+2?")

        assistant_msg = agent.state.messages[1]
        assert len(assistant_msg.content) == 2
        assert isinstance(assistant_msg.content[0], ThinkingContent)
        assert assistant_msg.content[0].thinking == "step by step"
        assert isinstance(assistant_msg.content[1], TextContent)
        assert assistant_msg.content[1].text == "4"


class TestContinue:
    @pytest.mark.asyncio
    async def test_throws_when_no_messages(self):
        faux = _faux()
        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="Test",
                model=faux.get_model(),
            ),
            stream_fn=stream_simple,
        ))

        with pytest.raises(RuntimeError, match="No messages"):
            await agent.continue_()

    @pytest.mark.asyncio
    async def test_continues_from_user_message(self):
        faux = _faux()
        faux.set_responses([faux_assistant_message("HELLO WORLD")])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="You are a helpful assistant.",
                model=faux.get_model(),
                thinking_level="off",
            ),
            stream_fn=stream_simple,
        ))

        agent.state.messages = [
            UserMessage(content="Say HELLO WORLD", timestamp=_now()),
        ]
        await agent.continue_()

        assert agent.state.is_streaming is False
        assert len(agent.state.messages) == 2
        assert agent.state.messages[1].role == "assistant"
        assert "HELLO WORLD" in agent.state.messages[1].content[0].text

    @pytest.mark.asyncio
    async def test_continues_from_tool_result(self):
        from pi_llm import AssistantMessage, ToolResultMessage, Usage

        faux = _faux()
        model = faux.get_model()
        faux.set_responses([faux_assistant_message("The answer is 8.")])

        agent = Agent(AgentOptions(
            initial_state=InitialAgentState(
                system_prompt="State the answer clearly.",
                model=model,
                thinking_level="off",
                tools=[CalculateTool()],
            ),
            stream_fn=stream_simple,
        ))

        agent.state.messages = [
            UserMessage(content="What is 5+3?", timestamp=_now()),
            AssistantMessage(
                content=[
                    TextContent(text="Let me calculate."),
                    ToolCall(id="calc-1", name="calculate", arguments={"expression": "5+3"}),
                ],
                api=model.api, provider=model.provider, model=model.id,
                usage=Usage(), stop_reason="toolUse", timestamp=_now(),
            ),
            ToolResultMessage(
                tool_call_id="calc-1", tool_name="calculate",
                content=[TextContent(text="5+3 = 8")],
                is_error=False, timestamp=_now(),
            ),
        ]

        await agent.continue_()

        assert len(agent.state.messages) >= 4
        last = agent.state.messages[-1]
        assert last.role == "assistant"
        assert "8" in last.content[0].text
