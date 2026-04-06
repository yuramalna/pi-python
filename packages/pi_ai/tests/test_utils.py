"""Tests for pi_ai utility functions."""

import pytest

from pi_ai import (
    AssistantMessage,
    Tool,
    ToolCall,
    Usage,
    is_context_overflow,
    parse_streaming_json,
    sanitize_surrogates,
    short_hash,
    validate_tool_arguments,
    validate_tool_call,
)

# -- parse_streaming_json --


def test_parse_streaming_json_complete():
    assert parse_streaming_json('{"key": "value"}') == {"key": "value"}


def test_parse_streaming_json_partial():
    result = parse_streaming_json('{"key": "val')
    assert result == {"key": "val"}


def test_parse_streaming_json_empty():
    assert parse_streaming_json("") == {}


def test_parse_streaming_json_garbage():
    assert parse_streaming_json("garbage") == {}


def test_parse_streaming_json_whitespace():
    assert parse_streaming_json("   ") == {}


# -- validate_tool_call / validate_tool_arguments --

_TOOL = Tool(
    name="search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
)


def test_validate_tool_call_valid():
    tc = ToolCall(id="1", name="search", arguments={"q": "hello"})
    assert validate_tool_call([_TOOL], tc) == {"q": "hello"}


def test_validate_tool_call_missing_tool():
    tc = ToolCall(id="1", name="nonexistent", arguments={})
    with pytest.raises(ValueError, match='Tool "nonexistent" not found'):
        validate_tool_call([_TOOL], tc)


def test_validate_tool_arguments_invalid():
    tc = ToolCall(id="1", name="search", arguments={})
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(_TOOL, tc)


def test_validate_tool_arguments_deepcopy():
    original_args = {"q": "hello"}
    tc = ToolCall(id="1", name="search", arguments=original_args)
    result = validate_tool_arguments(_TOOL, tc)
    result["q"] = "modified"
    assert tc.arguments["q"] == "hello"  # original unchanged


# -- is_context_overflow --


def test_is_context_overflow_error_pattern():
    msg = AssistantMessage(
        stop_reason="error",
        error_message="prompt is too long: 200000 tokens > 100000 maximum",
        timestamp=0,
    )
    assert is_context_overflow(msg) is True


def test_is_context_overflow_openai_pattern():
    msg = AssistantMessage(
        stop_reason="error",
        error_message="Your input exceeds the context window of this model",
        timestamp=0,
    )
    assert is_context_overflow(msg) is True


def test_is_context_overflow_non_overflow_error():
    msg = AssistantMessage(
        stop_reason="error",
        error_message="Throttling error: Too many tokens, please wait",
        timestamp=0,
    )
    assert is_context_overflow(msg) is False


def test_is_context_overflow_rate_limit():
    msg = AssistantMessage(
        stop_reason="error",
        error_message="rate limit exceeded",
        timestamp=0,
    )
    assert is_context_overflow(msg) is False


def test_is_context_overflow_silent():
    msg = AssistantMessage(
        stop_reason="stop",
        usage=Usage(input=200000, cache_read=10000),
        timestamp=0,
    )
    assert is_context_overflow(msg, context_window=128000) is True


def test_is_context_overflow_normal():
    msg = AssistantMessage(stop_reason="stop", timestamp=0)
    assert is_context_overflow(msg) is False


# -- short_hash --


def test_short_hash_deterministic():
    assert short_hash("test") == short_hash("test")


def test_short_hash_different_inputs():
    assert short_hash("a") != short_hash("b")


def test_short_hash_length():
    assert len(short_hash("anything")) == 12


# -- sanitize_surrogates --


def test_sanitize_surrogates_normal_text():
    assert sanitize_surrogates("Hello World") == "Hello World"
    assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"


def test_sanitize_surrogates_empty():
    assert sanitize_surrogates("") == ""
