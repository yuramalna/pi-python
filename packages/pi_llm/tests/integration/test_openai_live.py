"""Integration tests for pi_llm with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

import pytest

from pi_llm.models import get_model
from pi_llm.stream import complete_simple, stream_simple
from pi_llm.types import (
    Context,
    SimpleStreamOptions,
    Tool,
    ToolCall,
    UserMessage,
)

MODEL_ID = "gpt-5.4-2026-03-05"


@pytest.mark.integration
async def test_complete_simple_text(openai_api_key):
    """Basic text completion returns a valid AssistantMessage."""
    model = get_model("openai", MODEL_ID)
    ctx = Context(
        system_prompt="Reply in one word.",
        messages=[UserMessage(content="Say hello", timestamp=0)],
    )
    result = await complete_simple(model, ctx, SimpleStreamOptions(api_key=openai_api_key))

    assert result.stop_reason == "stop"
    assert len(result.content) > 0
    assert result.content[0].type == "text"
    assert result.usage.total_tokens > 0


@pytest.mark.integration
async def test_stream_simple_events(openai_api_key):
    """Streaming produces the expected event sequence: start, text_delta(s), done."""
    model = get_model("openai", MODEL_ID)
    ctx = Context(
        system_prompt="Reply in one word.",
        messages=[UserMessage(content="Say hello", timestamp=0)],
    )
    events = []
    async for event in stream_simple(model, ctx, SimpleStreamOptions(api_key=openai_api_key)):
        events.append(event)

    types = [e.type for e in events]
    assert "start" in types
    assert "text_delta" in types
    assert "done" in types

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    assert len(done_events[0].message.content) > 0


@pytest.mark.integration
async def test_tool_call(openai_api_key):
    """Model invokes a tool when instructed."""
    tool = Tool(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    )
    model = get_model("openai", MODEL_ID)
    ctx = Context(
        system_prompt="You must use the get_weather tool to answer weather questions. Do not answer without calling the tool.",
        messages=[UserMessage(content="What's the weather in Paris?", timestamp=0)],
        tools=[tool],
    )
    result = await complete_simple(model, ctx, SimpleStreamOptions(api_key=openai_api_key))

    assert result.stop_reason == "toolUse"
    tool_calls = [c for c in result.content if isinstance(c, ToolCall)]
    assert len(tool_calls) > 0
    assert tool_calls[0].name == "get_weather"
