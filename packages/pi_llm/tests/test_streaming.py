"""Tests for the streaming pipeline using the faux provider."""

import asyncio

import pytest

from pi_llm import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
    complete,
    complete_simple,
    stream,
    stream_simple,
)
from pi_llm.providers.faux import (
    RegisterFauxProviderOptions,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
    register_faux_provider,
)

_registrations: list = []


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for reg in _registrations:
        reg.unregister()
    _registrations.clear()


def _reg(**kw):
    r = register_faux_provider(RegisterFauxProviderOptions(**kw) if kw else None)
    _registrations.append(r)
    return r


def _ctx(text="hi"):
    return Context(messages=[UserMessage(content=text, timestamp=1)])


class TestStreamFunction:
    @pytest.mark.asyncio
    async def test_stream_returns_async_iterable(self):
        reg = _reg()
        reg.set_responses([faux_assistant_message("hello")])

        events = []
        async for event in stream(reg.get_model(), _ctx()):
            events.append(event)

        assert len(events) >= 3  # start, text_start, text_delta, text_end, done
        assert events[0].type == "start"
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_result_returns_final_message(self):
        reg = _reg()
        reg.set_responses([faux_assistant_message("result text")])

        s = stream(reg.get_model(), _ctx())
        result = await s.result()

        assert isinstance(result, AssistantMessage)
        assert result.content[0].text == "result text"
        assert result.stop_reason == "stop"


class TestCompleteFunction:
    @pytest.mark.asyncio
    async def test_complete_returns_message(self):
        reg = _reg()
        reg.set_responses([faux_assistant_message("completed")])

        result = await complete(reg.get_model(), _ctx())
        assert result.content == [TextContent(text="completed")]

    @pytest.mark.asyncio
    async def test_complete_simple_returns_message(self):
        reg = _reg()
        reg.set_responses([faux_assistant_message("simple")])

        result = await complete_simple(reg.get_model(), _ctx())
        assert result.content == [TextContent(text="simple")]


class TestStreamSimpleFunction:
    @pytest.mark.asyncio
    async def test_stream_simple_with_reasoning(self):
        reg = _reg()
        reg.set_responses([
            faux_assistant_message([faux_thinking("let me think"), faux_text("answer")]),
        ])

        events = []
        async for event in stream_simple(
            reg.get_model(), _ctx(),
            SimpleStreamOptions(reasoning="medium"),
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types
        assert "text_start" in types
        assert "done" in types


class TestMultiTurnConversation:
    @pytest.mark.asyncio
    async def test_multi_turn_preserves_context(self):
        reg = _reg()
        reg.set_responses([
            lambda ctx, opts, state, model: faux_assistant_message(
                f"turn {state['callCount']}, msgs={len(ctx.messages)}"
            ),
            lambda ctx, opts, state, model: faux_assistant_message(
                f"turn {state['callCount']}, msgs={len(ctx.messages)}"
            ),
        ])

        ctx = _ctx("first message")
        r1 = await complete(reg.get_model(), ctx)
        assert r1.content[0].text == "turn 1, msgs=1"

        ctx.messages.append(r1)
        ctx.messages.append(UserMessage(content="second message", timestamp=2))
        r2 = await complete(reg.get_model(), ctx)
        assert r2.content[0].text == "turn 2, msgs=3"


class TestToolCallStreaming:
    @pytest.mark.asyncio
    async def test_tool_use_response(self):
        reg = _reg()
        reg.set_responses([
            faux_assistant_message(
                [faux_tool_call("search", {"query": "test"}, id="tc-1")],
                stop_reason="toolUse",
            ),
        ])

        result = await complete(reg.get_model(), _ctx())
        assert result.stop_reason == "toolUse"
        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolCall)
        assert result.content[0].name == "search"
        assert result.content[0].arguments == {"query": "test"}

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self):
        reg = _reg()
        reg.set_responses([
            faux_assistant_message(
                [
                    faux_text("I'll search for that"),
                    faux_tool_call("search", {"q": "test"}, id="tc-1"),
                    faux_tool_call("search", {"q": "test2"}, id="tc-2"),
                ],
                stop_reason="toolUse",
            ),
        ])

        result = await complete(reg.get_model(), _ctx())
        assert len(result.content) == 3
        assert isinstance(result.content[0], TextContent)
        assert isinstance(result.content[1], ToolCall)
        assert isinstance(result.content[2], ToolCall)


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_error_response(self):
        reg = _reg()
        reg.set_responses([
            faux_assistant_message("error text", stop_reason="error", error_message="Something went wrong"),
        ])

        result = await complete(reg.get_model(), _ctx())
        assert result.stop_reason == "error"
        assert result.error_message == "Something went wrong"

    @pytest.mark.asyncio
    async def test_no_provider_raises(self):
        with pytest.raises(ValueError, match="No API provider registered"):
            from pi_llm.types import Model
            fake_model = Model(id="x", name="x", api="nonexistent-api", provider="x")
            await complete(fake_model, _ctx())

    @pytest.mark.asyncio
    async def test_exhausted_responses(self):
        reg = _reg()
        # No responses queued
        result = await complete(reg.get_model(), _ctx())
        assert result.stop_reason == "error"
        assert "No more faux responses" in result.error_message
