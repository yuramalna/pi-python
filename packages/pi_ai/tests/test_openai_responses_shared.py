"""Tests for OpenAI Responses message conversion, tools, and stream processing."""

import json
from types import SimpleNamespace

import pytest

from pi_ai.providers.openai_responses_shared import (
    _build_foreign_responses_item_id,
    _normalize_id_part,
    convert_responses_messages,
    convert_responses_tools,
    encode_text_signature_v1,
    map_stop_reason,
    parse_text_signature,
    process_responses_stream,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream


def _model(*, id="gpt-4o", provider="openai", api="openai-responses", reasoning=False):
    return Model(id=id, name=id, api=api, provider=provider, reasoning=reasoning)


# =============================================================================
# TextSignatureV1
# =============================================================================


def test_encode_text_signature_v1():
    sig = encode_text_signature_v1("msg_123")
    parsed = json.loads(sig)
    assert parsed == {"v": 1, "id": "msg_123"}


def test_encode_text_signature_v1_with_phase():
    sig = encode_text_signature_v1("msg_123", phase="commentary")
    parsed = json.loads(sig)
    assert parsed == {"v": 1, "id": "msg_123", "phase": "commentary"}


def test_parse_text_signature_v1():
    result = parse_text_signature('{"v": 1, "id": "msg_123", "phase": "final_answer"}')
    assert result == {"id": "msg_123", "phase": "final_answer"}


def test_parse_text_signature_v1_no_phase():
    result = parse_text_signature('{"v": 1, "id": "msg_123"}')
    assert result == {"id": "msg_123"}


def test_parse_text_signature_legacy():
    result = parse_text_signature("plain_string_id")
    assert result == {"id": "plain_string_id"}


def test_parse_text_signature_none():
    assert parse_text_signature(None) is None
    assert parse_text_signature("") is None


# =============================================================================
# ID normalization
# =============================================================================


def test_normalize_id_part_special_chars():
    assert _normalize_id_part("abc!@#def") == "abc___def"


def test_normalize_id_part_truncate():
    long_id = "a" * 100
    assert len(_normalize_id_part(long_id)) == 64


def test_normalize_id_part_trailing_underscores():
    assert _normalize_id_part("abc___") == "abc"


def test_build_foreign_item_id():
    result = _build_foreign_responses_item_id("some_item_id")
    assert result.startswith("fc_")
    assert len(result) <= 64


# =============================================================================
# convert_responses_messages
# =============================================================================


def test_system_prompt_developer_for_reasoning():
    model = _model(reasoning=True)
    ctx = Context(system_prompt="You are helpful.", messages=[])
    msgs = convert_responses_messages(model, ctx, set())
    assert msgs[0]["role"] == "developer"


def test_system_prompt_system_for_non_reasoning():
    model = _model()
    ctx = Context(system_prompt="You are helpful.", messages=[])
    msgs = convert_responses_messages(model, ctx, set())
    assert msgs[0]["role"] == "system"


def test_no_system_prompt_when_empty():
    ctx = Context(system_prompt="", messages=[])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert len(msgs) == 0


def test_user_string_content():
    ctx = Context(messages=[UserMessage(content="Hello", timestamp=0)])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["type"] == "input_text"
    assert msgs[0]["content"][0]["text"] == "Hello"


def test_user_with_image():
    model = _model()  # default input_types=["text"]
    ctx = Context(messages=[UserMessage(
        content=[
            TextContent(text="look"),
            ImageContent(data="abc123", mime_type="image/png"),
        ],
        timestamp=0,
    )])
    msgs = convert_responses_messages(model, ctx, set())
    # Images filtered out since model doesn't declare "image" in input_types
    assert len(msgs[0]["content"]) == 1
    assert msgs[0]["content"][0]["type"] == "input_text"


def test_user_with_image_supported():
    model = Model(id="gpt-4o", name="gpt-4o", api="openai-responses", provider="openai", input_types=["text", "image"])
    ctx = Context(messages=[UserMessage(
        content=[
            TextContent(text="look"),
            ImageContent(data="abc123", mime_type="image/png"),
        ],
        timestamp=0,
    )])
    msgs = convert_responses_messages(model, ctx, set())
    assert len(msgs[0]["content"]) == 2
    assert msgs[0]["content"][1]["type"] == "input_image"


def test_assistant_text():
    assistant = AssistantMessage(
        content=[TextContent(text="Hi there")],
        provider="openai", api="openai-responses", model="gpt-4o", timestamp=0,
    )
    ctx = Context(messages=[assistant])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert msgs[0]["type"] == "message"
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"][0]["text"] == "Hi there"


def test_assistant_thinking_with_signature():
    sig = json.dumps({"type": "reasoning", "id": "rs_123", "summary": [{"type": "summary_text", "text": "thought"}]})
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking="thought", thinking_signature=sig)],
        provider="openai", api="openai-responses", model="gpt-4o", timestamp=0,
    )
    ctx = Context(messages=[assistant])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert msgs[0]["type"] == "reasoning"
    assert msgs[0]["id"] == "rs_123"


def test_assistant_thinking_signature_strips_null_status():
    """Regression: reasoning items with status=null must not be replayed verbatim."""
    sig = json.dumps({
        "type": "reasoning", "id": "rs_123",
        "summary": [{"type": "summary_text", "text": "thought"}],
        "encrypted_content": "gAAAAfake",
        "status": None,
    })
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking="thought", thinking_signature=sig)],
        provider="openai", api="openai-responses", model="gpt-4o", timestamp=0,
    )
    ctx = Context(messages=[assistant])
    msgs = convert_responses_messages(_model(), ctx, set())
    reasoning = msgs[0]
    assert reasoning["type"] == "reasoning"
    assert "status" not in reasoning, "status=null must be stripped to avoid OpenAI 400 error"


def test_assistant_thinking_signature_preserves_valid_status():
    """Valid status values like 'completed' should be preserved on replay."""
    sig = json.dumps({
        "type": "reasoning", "id": "rs_456",
        "summary": [{"type": "summary_text", "text": "done"}],
        "encrypted_content": "gAAAAfake",
        "status": "completed",
    })
    assistant = AssistantMessage(
        content=[ThinkingContent(thinking="done", thinking_signature=sig)],
        provider="openai", api="openai-responses", model="gpt-4o", timestamp=0,
    )
    ctx = Context(messages=[assistant])
    msgs = convert_responses_messages(_model(), ctx, set())
    reasoning = msgs[0]
    assert reasoning["status"] == "completed"


def test_assistant_tool_call_compound_id():
    tc = ToolCall(id="call_abc|fc_def", name="get_weather", arguments={"city": "LA"})
    assistant = AssistantMessage(
        content=[tc],
        provider="openai", api="openai-responses", model="gpt-4o", timestamp=0,
    )
    ctx = Context(messages=[assistant])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert msgs[0]["type"] == "function_call"
    assert msgs[0]["call_id"] == "call_abc"
    assert msgs[0]["id"] == "fc_def"
    assert msgs[0]["arguments"] == '{"city": "LA"}'


def test_tool_result_maps_to_function_call_output():
    tr = ToolResultMessage(
        tool_call_id="call_abc|fc_def",
        tool_name="get_weather",
        content=[TextContent(text="Sunny, 72F")],
        timestamp=0,
    )
    ctx = Context(messages=[tr])
    msgs = convert_responses_messages(_model(), ctx, set())
    assert msgs[0]["type"] == "function_call_output"
    assert msgs[0]["call_id"] == "call_abc"
    assert msgs[0]["output"] == "Sunny, 72F"


# =============================================================================
# convert_responses_tools
# =============================================================================


def test_tool_conversion():
    tools = [Tool(name="get_weather", description="Get weather", parameters={"type": "object"})]
    result = convert_responses_tools(tools)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "get_weather"
    assert result[0]["strict"] is False


def test_tool_conversion_strict():
    tools = [Tool(name="fn", description="d", parameters={})]
    result = convert_responses_tools(tools, strict=True)
    assert result[0]["strict"] is True


# =============================================================================
# map_stop_reason
# =============================================================================


def test_map_stop_reason_all_statuses():
    assert map_stop_reason("completed") == "stop"
    assert map_stop_reason("incomplete") == "length"
    assert map_stop_reason("failed") == "error"
    assert map_stop_reason("cancelled") == "error"
    assert map_stop_reason("in_progress") == "stop"
    assert map_stop_reason("queued") == "stop"
    assert map_stop_reason(None) == "stop"


# =============================================================================
# process_responses_stream (mock events)
# =============================================================================


async def _mock_stream(events):
    """Async generator from a list of SimpleNamespace events."""
    for e in events:
        yield e


def _ns(**kwargs):
    """Shorthand for SimpleNamespace."""
    return SimpleNamespace(**kwargs)


async def test_stream_text_events():
    events = [
        _ns(type="response.created", response=_ns(id="resp_1")),
        _ns(type="response.output_item.added", item=_ns(type="message", content=[])),
        _ns(type="response.content_part.added", part=_ns(type="output_text", text="")),
        _ns(type="response.output_text.delta", delta="Hello"),
        _ns(type="response.output_text.delta", delta=" world"),
        _ns(type="response.output_item.done", item=_ns(
            type="message", id="msg_1", content=[_ns(type="output_text", text="Hello world")], phase=None,
        )),
        _ns(type="response.completed", response=_ns(
            id="resp_1", status="completed", service_tier=None,
            usage=_ns(
                input_tokens=10, output_tokens=5, total_tokens=15,
                input_tokens_details=_ns(cached_tokens=0),
            ),
        )),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    await process_responses_stream(_mock_stream(events), output, stream, _model())

    assert output.response_id == "resp_1"
    assert len(output.content) == 1
    assert output.content[0].type == "text"
    assert output.content[0].text == "Hello world"
    assert output.stop_reason == "stop"
    assert output.usage.input == 10
    assert output.usage.output == 5


async def test_stream_tool_call_events():
    events = [
        _ns(type="response.created", response=_ns(id="resp_1")),
        _ns(type="response.output_item.added", item=_ns(
            type="function_call", call_id="call_1", id="fc_1", name="get_weather", arguments="",
        )),
        _ns(type="response.function_call_arguments.delta", delta='{"ci'),
        _ns(type="response.function_call_arguments.delta", delta='ty":"LA"}'),
        _ns(type="response.function_call_arguments.done", arguments='{"city":"LA"}'),
        _ns(type="response.output_item.done", item=_ns(
            type="function_call", call_id="call_1", id="fc_1", name="get_weather", arguments='{"city":"LA"}',
        )),
        _ns(type="response.completed", response=_ns(
            id="resp_1", status="completed", service_tier=None,
            usage=_ns(input_tokens=10, output_tokens=5, total_tokens=15, input_tokens_details=_ns(cached_tokens=0)),
        )),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    await process_responses_stream(_mock_stream(events), output, stream, _model())

    assert len(output.content) == 1
    assert output.content[0].type == "toolCall"
    assert output.stop_reason == "toolUse"


async def test_stream_thinking_events():
    events = [
        _ns(type="response.created", response=_ns(id="resp_1")),
        _ns(type="response.output_item.added", item=_ns(type="reasoning")),
        _ns(type="response.reasoning_summary_part.added", part=_ns(type="summary_text", text="")),
        _ns(type="response.reasoning_summary_text.delta", delta="Thinking..."),
        _ns(type="response.reasoning_summary_part.done", part=_ns(type="summary_text", text="Thinking...")),
        _ns(type="response.output_item.done", item=_ns(
            type="reasoning", id="rs_1",
            summary=[_ns(type="summary_text", text="Thinking...\n\n")],
        )),
        _ns(type="response.completed", response=_ns(
            id="resp_1", status="completed", service_tier=None,
            usage=_ns(input_tokens=10, output_tokens=5, total_tokens=15, input_tokens_details=_ns(cached_tokens=0)),
        )),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    await process_responses_stream(_mock_stream(events), output, stream, _model())

    assert len(output.content) == 1
    assert output.content[0].type == "thinking"
    assert output.content[0].thinking_signature is not None


async def test_stream_error_raises():
    events = [
        _ns(type="error", code="server_error", message="Something went wrong"),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    with pytest.raises(RuntimeError, match="Error Code server_error"):
        await process_responses_stream(_mock_stream(events), output, stream, _model())


async def test_stream_failed_raises():
    events = [
        _ns(type="response.failed", response=_ns(
            error=_ns(code="rate_limit", message="Too many requests"),
            incomplete_details=None,
        )),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    with pytest.raises(RuntimeError, match="rate_limit"):
        await process_responses_stream(_mock_stream(events), output, stream, _model())


async def test_stream_usage_subtracts_cached_tokens():
    events = [
        _ns(type="response.completed", response=_ns(
            id="resp_1", status="completed", service_tier=None,
            usage=_ns(
                input_tokens=100, output_tokens=50, total_tokens=150,
                input_tokens_details=_ns(cached_tokens=30),
            ),
        )),
    ]

    output = AssistantMessage(timestamp=0)
    stream = AssistantMessageEventStream()
    await process_responses_stream(_mock_stream(events), output, stream, _model())

    assert output.usage.input == 70  # 100 - 30
    assert output.usage.cache_read == 30
    assert output.usage.output == 50
