"""Tests for the faux (mock) LLM provider."""

import asyncio
import json
import math

import pytest

from pi_llm import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    UserMessage,
    complete,
    stream,
)
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


async def collect_events(event_stream) -> list[AssistantMessageEvent]:
    events = []
    async for event in event_stream:
        events.append(event)
    return events


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _make_context(text: str = "hi") -> Context:
    return Context(messages=[UserMessage(content=text, timestamp=_now_ms())])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_registrations: list = []


@pytest.fixture(autouse=True)
def _cleanup_registrations():
    yield
    for reg in _registrations:
        reg.unregister()
    _registrations.clear()


def _register(**kwargs):
    reg = register_faux_provider(
        RegisterFauxProviderOptions(**kwargs) if kwargs else None
    )
    _registrations.append(reg)
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistrationAndModels:
    @pytest.mark.asyncio
    async def test_registers_provider_and_estimates_usage(self):
        reg = _register()
        reg.set_responses([faux_assistant_message("hello world")])

        ctx = Context(
            system_prompt="Be concise.",
            messages=[UserMessage(content="hi there", timestamp=_now_ms())],
        )
        response = await complete(reg.get_model(), ctx)

        assert response.content == [TextContent(text="hello world")]
        assert response.usage.input > 0
        assert response.usage.output > 0
        assert response.usage.total_tokens == (
            response.usage.input + response.usage.output
        )
        assert reg.state["callCount"] == 1

    @pytest.mark.asyncio
    async def test_supports_multiple_models(self):
        reg = _register(
            models=[
                FauxModelDefinition(id="faux-fast", name="Faux Fast", reasoning=False),
                FauxModelDefinition(id="faux-thinker", name="Faux Thinker", reasoning=True),
            ],
        )
        reg.set_responses([
            lambda ctx, opts, state, model: faux_assistant_message(
                f"{model.id}:{model.reasoning}"
            ),
            lambda ctx, opts, state, model: faux_assistant_message(
                f"{model.id}:{model.reasoning}"
            ),
        ])

        assert [m.id for m in reg.models] == ["faux-fast", "faux-thinker"]
        assert reg.get_model() is reg.models[0]
        assert reg.get_model("faux-fast").reasoning is False
        assert reg.get_model("faux-thinker").reasoning is True

        fast = await complete(reg.get_model("faux-fast"), _make_context())
        thinker = await complete(reg.get_model("faux-thinker"), _make_context())

        assert fast.content == [TextContent(text="faux-fast:False")]
        assert thinker.content == [TextContent(text="faux-thinker:True")]

    @pytest.mark.asyncio
    async def test_rewrites_api_provider_model(self):
        reg = _register(
            api="faux:test",
            provider="faux-provider",
            models=[FauxModelDefinition(id="faux-model")],
        )
        reg.set_responses([faux_assistant_message("hello")])

        response = await complete(reg.get_model(), _make_context())

        assert response.api == "faux:test"
        assert response.provider == "faux-provider"
        assert response.model == "faux-model"

    @pytest.mark.asyncio
    async def test_unregisters_provider(self):
        reg = _register()
        reg.set_responses([faux_assistant_message("hello")])
        reg.unregister()
        _registrations.remove(reg)

        with pytest.raises(ValueError, match="No API provider registered"):
            await complete(reg.get_model(), _make_context())


class TestResponseQueue:
    @pytest.mark.asyncio
    async def test_consumes_in_order_and_errors_when_exhausted(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])
        ctx = _make_context()

        first = await complete(reg.get_model(), ctx)
        second = await complete(reg.get_model(), ctx)
        exhausted = await complete(reg.get_model(), ctx)

        assert first.content == [TextContent(text="first")]
        assert second.content == [TextContent(text="second")]
        assert exhausted.stop_reason == "error"
        assert exhausted.error_message == "No more faux responses queued"
        assert reg.get_pending_response_count() == 0
        assert reg.state["callCount"] == 3

    @pytest.mark.asyncio
    async def test_replace_and_append_responses(self):
        reg = _register()
        ctx = _make_context()

        reg.set_responses([faux_assistant_message("first")])
        r1 = await complete(reg.get_model(), ctx)
        assert r1.content == [TextContent(text="first")]
        assert reg.get_pending_response_count() == 0

        reg.set_responses([faux_assistant_message("second")])
        assert reg.get_pending_response_count() == 1
        r2 = await complete(reg.get_model(), ctx)
        assert r2.content == [TextContent(text="second")]

        reg.append_responses([
            faux_assistant_message("third"),
            faux_assistant_message("fourth"),
        ])
        assert reg.get_pending_response_count() == 2
        r3 = await complete(reg.get_model(), ctx)
        r4 = await complete(reg.get_model(), ctx)
        assert r3.content == [TextContent(text="third")]
        assert r4.content == [TextContent(text="fourth")]
        assert reg.get_pending_response_count() == 0


class TestContentBlocks:
    @pytest.mark.asyncio
    async def test_thinking_text_and_tool_call_blocks(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message(
                [
                    faux_thinking("think"),
                    faux_tool_call("echo", {"text": "hi"}),
                    faux_text("done"),
                ],
                stop_reason="toolUse",
            ),
        ])

        response = await complete(reg.get_model(), _make_context())

        assert len(response.content) == 3
        assert isinstance(response.content[0], ThinkingContent)
        assert response.content[0].thinking == "think"
        assert isinstance(response.content[1], ToolCall)
        assert response.content[1].name == "echo"
        assert response.content[1].arguments == {"text": "hi"}
        assert isinstance(response.content[2], TextContent)
        assert response.content[2].text == "done"
        assert response.stop_reason == "toolUse"


class TestFactoryResponses:
    @pytest.mark.asyncio
    async def test_sync_factory(self):
        reg = _register()
        reg.set_responses([
            lambda ctx, opts, state, model: faux_assistant_message(
                f"{len(ctx.messages)}:{state['callCount']}"
            ),
        ])

        response = await complete(reg.get_model(), _make_context())
        assert response.content == [TextContent(text="1:1")]

    @pytest.mark.asyncio
    async def test_async_factory(self):
        reg = _register()

        async def factory(ctx, opts, state, model):
            return faux_assistant_message(
                f"async:{len(ctx.messages)}:{state['callCount']}"
            )

        reg.set_responses([factory])

        response = await complete(reg.get_model(), _make_context())
        assert response.content == [TextContent(text="async:1:1")]

    @pytest.mark.asyncio
    async def test_factory_that_throws(self):
        reg = _register()
        reg.set_responses([
            lambda ctx, opts, state, model: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        ])

        events = await collect_events(
            stream(reg.get_model(), _make_context())
        )

        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].error.stop_reason == "error"
        assert "boom" in events[0].error.error_message


class TestUsageEstimation:
    @pytest.mark.asyncio
    async def test_estimates_prompt_and_output_tokens(self):
        reg = _register()
        reg.set_responses([faux_assistant_message("done")])

        tool = Tool(
            name="echo",
            description="Echo back text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        ctx = Context(
            system_prompt="sys",
            messages=[
                UserMessage(
                    content=[
                        TextContent(text="hello"),
                    ],
                    timestamp=1,
                ),
            ],
            tools=[tool],
        )

        response = await complete(reg.get_model(), ctx)

        assert response.usage.input > 0
        assert response.usage.output > 0
        assert response.usage.cache_read == 0
        assert response.usage.cache_write == 0
        assert response.usage.total_tokens == (
            response.usage.input + response.usage.output
        )

    @pytest.mark.asyncio
    async def test_simulates_prompt_caching_per_session(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])

        ctx = Context(
            system_prompt="Be concise.",
            messages=[UserMessage(content="hello", timestamp=_now_ms())],
        )

        first = await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-1", cache_retention="short"),
        )
        assert first.usage.cache_read == 0
        assert first.usage.cache_write > 0

        # Extend conversation
        ctx.messages.append(first)
        ctx.messages.append(UserMessage(content="follow up", timestamp=_now_ms() + 1))

        second = await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-1", cache_retention="short"),
        )
        assert second.usage.cache_read > 0

    @pytest.mark.asyncio
    async def test_no_cache_across_different_sessions(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])

        ctx = Context(
            messages=[UserMessage(content="hello", timestamp=_now_ms())],
        )

        first = await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-1", cache_retention="short"),
        )
        ctx.messages.append(first)
        ctx.messages.append(UserMessage(content="follow up", timestamp=_now_ms() + 1))

        second = await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-2", cache_retention="short"),
        )
        assert second.usage.cache_read == 0
        assert second.usage.cache_write > 0

    @pytest.mark.asyncio
    async def test_no_caching_when_retention_is_none(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message("first"),
            faux_assistant_message("second"),
        ])

        ctx = Context(
            messages=[UserMessage(content="hello", timestamp=_now_ms())],
        )

        await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-1", cache_retention="none"),
        )
        ctx.messages.append(faux_assistant_message("first"))
        ctx.messages.append(UserMessage(content="follow up", timestamp=_now_ms() + 1))

        second = await complete(
            reg.get_model(), ctx,
            SimpleStreamOptions(session_id="session-1", cache_retention="none"),
        )
        assert second.usage.cache_read == 0
        assert second.usage.cache_write == 0


class TestStreaming:
    @pytest.mark.asyncio
    async def test_streams_thinking_text_and_tool_call_deltas(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message(
                [
                    faux_thinking("thinking text"),
                    faux_text("answer text"),
                    faux_tool_call("echo", {"text": "hi", "count": 12}, id="tool-1"),
                ],
                stop_reason="toolUse",
            ),
        ])

        event_types = []
        tool_call_deltas = []
        s = stream(reg.get_model(), _make_context())
        async for event in s:
            event_types.append(event.type)
            if event.type == "toolcall_delta":
                tool_call_deltas.append(event.delta)

        assert "thinking_start" in event_types
        assert "thinking_delta" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "toolcall_start" in event_types
        assert "toolcall_delta" in event_types
        assert "toolcall_end" in event_types
        assert len(tool_call_deltas) >= 1
        assert json.loads("".join(tool_call_deltas)) == {"text": "hi", "count": 12}

    @pytest.mark.asyncio
    async def test_exact_event_order_fixed_chunks(self):
        reg = _register(token_size_min=1, token_size_max=1)
        reg.set_responses([
            faux_assistant_message(
                [
                    faux_thinking("go"),
                    faux_text("ok"),
                    faux_tool_call("echo", {}, id="tool-1"),
                ],
                stop_reason="toolUse",
            ),
        ])

        events = await collect_events(stream(reg.get_model(), _make_context()))
        types = [e.type for e in events]

        assert types == [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]

    @pytest.mark.asyncio
    async def test_streams_multiple_tool_calls(self):
        reg = _register()
        reg.set_responses([
            faux_assistant_message(
                [
                    faux_tool_call("echo", {"text": "one"}, id="tool-1"),
                    faux_tool_call("echo", {"text": "two"}, id="tool-2"),
                ],
                stop_reason="toolUse",
            ),
        ])

        events = await collect_events(stream(reg.get_model(), _make_context()))
        starts = [e for e in events if e.type == "toolcall_start"]
        ends = [e for e in events if e.type == "toolcall_end"]

        assert len(starts) == 2
        assert len(ends) == 2


class TestErrorAndAbort:
    @pytest.mark.asyncio
    async def test_streams_error_message_as_terminal(self):
        reg = _register(token_size_min=2, token_size_max=2)
        msg = faux_assistant_message("partial", stop_reason="error", error_message="upstream failed")
        reg.set_responses([msg])

        events = await collect_events(stream(reg.get_model(), _make_context()))
        types = [e.type for e in events]

        assert types == ["start", "text_start", "text_delta", "text_end", "error"]
        terminal = events[-1]
        assert terminal.error.stop_reason == "error"
        assert terminal.error.error_message == "upstream failed"

    @pytest.mark.asyncio
    async def test_streams_aborted_message_as_terminal(self):
        reg = _register(token_size_min=2, token_size_max=2)
        msg = faux_assistant_message(
            "partial", stop_reason="aborted", error_message="Request was aborted",
        )
        reg.set_responses([msg])

        events = await collect_events(stream(reg.get_model(), _make_context()))
        types = [e.type for e in events]

        assert types == ["start", "text_start", "text_delta", "text_end", "error"]
        terminal = events[-1]
        assert terminal.error.stop_reason == "aborted"

    @pytest.mark.asyncio
    async def test_abort_before_first_chunk(self):
        reg = _register(tokens_per_second=50, token_size_min=3, token_size_max=3)
        reg.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

        cancel = asyncio.Event()
        cancel.set()  # already cancelled
        events = await collect_events(
            stream(
                reg.get_model(),
                _make_context(),
                StreamOptions(cancel_event=cancel),
            )
        )

        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].error.stop_reason == "aborted"

    @pytest.mark.asyncio
    async def test_abort_mid_text_stream(self):
        reg = _register(tokens_per_second=100, token_size_min=3, token_size_max=3)
        reg.set_responses([faux_assistant_message("abcdefghijklmnopqrstuvwxyz")])

        cancel = asyncio.Event()
        event_types = []
        text_delta_count = 0
        s = stream(
            reg.get_model(),
            _make_context(),
            StreamOptions(cancel_event=cancel),
        )
        async for event in s:
            event_types.append(event.type)
            if event.type == "text_delta":
                text_delta_count += 1
                cancel.set()

        assert text_delta_count == 1
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "error" in event_types
        assert "text_end" not in event_types

    @pytest.mark.asyncio
    async def test_abort_mid_thinking_stream(self):
        reg = _register(tokens_per_second=100, token_size_min=3, token_size_max=3)
        msg = faux_assistant_message("ignored")
        msg = msg.model_copy(update={
            "content": [ThinkingContent(thinking="abcdefghijklmnopqrstuvwxyz")],
        })
        reg.set_responses([msg])

        cancel = asyncio.Event()
        event_types = []
        thinking_delta_count = 0
        s = stream(
            reg.get_model(),
            _make_context(),
            StreamOptions(cancel_event=cancel),
        )
        async for event in s:
            event_types.append(event.type)
            if event.type == "thinking_delta":
                thinking_delta_count += 1
                cancel.set()

        assert thinking_delta_count == 1
        assert "thinking_start" in event_types
        assert "thinking_delta" in event_types
        assert "error" in event_types
        assert "thinking_end" not in event_types

    @pytest.mark.asyncio
    async def test_abort_mid_toolcall_stream(self):
        reg = _register(tokens_per_second=100, token_size_min=3, token_size_max=3)
        msg = faux_assistant_message("done", stop_reason="toolUse")
        msg = msg.model_copy(update={
            "content": [
                ToolCall(
                    id="tool-1",
                    name="echo",
                    arguments={"text": "abcdefghijklmnopqrstuvwxyz", "count": 123456789},
                ),
            ],
        })
        reg.set_responses([msg])

        cancel = asyncio.Event()
        event_types = []
        toolcall_delta_count = 0
        s = stream(
            reg.get_model(),
            _make_context(),
            StreamOptions(cancel_event=cancel),
        )
        async for event in s:
            event_types.append(event.type)
            if event.type == "toolcall_delta":
                toolcall_delta_count += 1
                cancel.set()

        assert toolcall_delta_count == 1
        assert "toolcall_start" in event_types
        assert "toolcall_delta" in event_types
        assert "error" in event_types
        assert "toolcall_end" not in event_types
