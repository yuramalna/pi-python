# pi-llm Quickstart

This guide walks you through your first LLM call with pi-llm: streaming text, collecting a complete response, and handling tool calls.

## Installation

```bash
pip install pi-llm
```

## Basic streaming

The simplest way to get started is `stream_simple`. It returns an async iterable of events that you consume with `async for`.

```python
import asyncio

from pi_llm import (
    Context,
    SimpleStreamOptions,
    TextDeltaEvent,
    get_model,
    stream_simple,
)
from pi_llm.providers import register_builtin_providers

# Register the built-in providers (OpenAI, etc.) once at startup
register_builtin_providers()


async def main():
    model = get_model("openai", "gpt-4o")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[],
    )

    event_stream = stream_simple(model, context, SimpleStreamOptions())

    async for event in event_stream:
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)

    print()  # newline after streaming finishes


asyncio.run(main())
```

!!! note
    `register_builtin_providers()` must be called once before any streaming function works. It registers the OpenAI Responses API provider (and any other built-in providers).

## Getting the complete message

If you do not need real-time streaming, use `complete_simple` to get the finished `AssistantMessage` directly:

```python
import asyncio

from pi_llm import (
    Context,
    SimpleStreamOptions,
    complete_simple,
    get_model,
)
from pi_llm.providers import register_builtin_providers

register_builtin_providers()


async def main():
    model = get_model("openai", "gpt-4o")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[],
    )

    message = await complete_simple(model, context, SimpleStreamOptions())

    # The message contains content blocks
    for block in message.content:
        if block.type == "text":
            print(block.text)


asyncio.run(main())
```

## Building a conversation

Add messages to the context to build a multi-turn conversation:

```python
import asyncio
import time

from pi_llm import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
    complete_simple,
    get_model,
)
from pi_llm.providers import register_builtin_providers

register_builtin_providers()


async def main():
    model = get_model("openai", "gpt-4o")
    messages = []

    # Turn 1
    messages.append(
        UserMessage(
            content="What is the capital of France?",
            timestamp=int(time.time() * 1000),
        )
    )
    context = Context(system_prompt="Answer concisely.", messages=messages)
    reply = await complete_simple(model, context, SimpleStreamOptions())
    messages.append(reply)

    # Turn 2
    messages.append(
        UserMessage(
            content="And what about Germany?",
            timestamp=int(time.time() * 1000),
        )
    )
    context = Context(system_prompt="Answer concisely.", messages=messages)
    reply = await complete_simple(model, context, SimpleStreamOptions())

    for block in reply.content:
        if block.type == "text":
            print(block.text)


asyncio.run(main())
```

## Streaming with tool calls

pi-llm supports tool use. Define a `Tool` with a JSON Schema, then handle `ToolCallEndEvent` in the event stream.

```python
import asyncio
import time

from pi_llm import (
    Context,
    DoneEvent,
    SimpleStreamOptions,
    TextContent,
    TextDeltaEvent,
    Tool,
    ToolCallEndEvent,
    ToolResultMessage,
    get_model,
    stream_simple,
)
from pi_llm.providers import register_builtin_providers

register_builtin_providers()

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)


async def main():
    model = get_model("openai", "gpt-4o")
    messages = []
    from pi_llm import UserMessage

    messages.append(
        UserMessage(
            content="What is the weather in Paris?",
            timestamp=int(time.time() * 1000),
        )
    )

    context = Context(
        system_prompt="Use the get_weather tool to answer weather questions.",
        messages=messages,
        tools=[weather_tool],
    )

    event_stream = stream_simple(model, context, SimpleStreamOptions())

    async for event in event_stream:
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)
        elif isinstance(event, ToolCallEndEvent):
            tool_call = event.tool_call
            print(f"\n[Tool call: {tool_call.name}({tool_call.arguments})]")

    # Get the assistant message that contains the tool call
    assistant_msg = await event_stream.result()
    messages.append(assistant_msg)

    # Simulate executing the tool and returning a result
    for block in assistant_msg.content:
        if block.type == "toolCall":
            messages.append(
                ToolResultMessage(
                    tool_call_id=block.id,
                    tool_name=block.name,
                    content=[TextContent(text='{"temp": "18C", "condition": "Sunny"}')],
                    timestamp=int(time.time() * 1000),
                )
            )

    # Continue the conversation with the tool result
    context = Context(
        system_prompt="Use the get_weather tool to answer weather questions.",
        messages=messages,
        tools=[weather_tool],
    )
    event_stream = stream_simple(model, context, SimpleStreamOptions())

    async for event in event_stream:
        if isinstance(event, TextDeltaEvent):
            print(event.delta, end="", flush=True)

    print()


asyncio.run(main())
```

## Enabling extended thinking

Models that support reasoning can be enabled with the `reasoning` option:

```python
from pi_llm import SimpleStreamOptions, ThinkingDeltaEvent

options = SimpleStreamOptions(reasoning="medium")

# In the event loop, you can observe thinking events:
# ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent
```

See [Thinking / Reasoning](concepts/thinking.md) for details on thinking levels.

## Next steps

- [Streaming](concepts/streaming.md) -- The four streaming functions and EventStream
- [Events](concepts/events.md) -- All 12 event types explained
- [Tools](concepts/tools.md) -- Tool definitions and JSON Schema
- [Messages & Context](concepts/messages-and-context.md) -- Message types and context serialization
- [Handle Tool Calls](howto/handle-tool-calls.md) -- Step-by-step tool call handling
