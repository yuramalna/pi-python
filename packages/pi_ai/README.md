# pi-ai

Unified LLM provider abstraction for Python. Provides streaming and non-streaming interfaces to LLM APIs with automatic model discovery, token and cost tracking, and tool call support.

## Features

- **Streaming & non-streaming** — `stream_simple()` / `complete_simple()` with async iteration
- **Model catalog** — `get_model()` with built-in pricing and metadata for OpenAI models
- **Tool calling** — Define tools with JSON Schema, validate arguments automatically
- **Extended thinking** — Unified reasoning level interface across providers
- **Cost tracking** — Automatic token counting and dollar cost calculation
- **Event-driven** — Fine-grained streaming events (text, thinking, tool calls)
- **Cross-provider types** — Pydantic models with camelCase/snake_case interop

## Supported Providers

- **OpenAI** (Responses API)

More providers coming soon.

## Installation

```bash
pip install pi-ai
```

## Quick Start

```python
import asyncio
from pi_ai import get_model, stream_simple, complete_simple, Context, UserMessage
from pi_ai import register_builtin_providers, SimpleStreamOptions

# Register providers (call once at startup)
register_builtin_providers()

model = get_model("openai", "gpt-4o")

context = Context(
    system_prompt="You are a helpful assistant.",
    messages=[UserMessage(content="What is Python?", timestamp=0)],
)

# Option 1: Non-streaming
async def main():
    message = await complete_simple(
        model, context,
        SimpleStreamOptions(api_key="sk-...")
    )
    for block in message.content:
        if hasattr(block, "text"):
            print(block.text)

asyncio.run(main())
```

```python
# Option 2: Streaming
async def main_stream():
    event_stream = stream_simple(
        model, context,
        SimpleStreamOptions(api_key="sk-...")
    )
    async for event in event_stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)

asyncio.run(main_stream())
```

## Tools

Define tools with JSON Schema and handle tool calls in a loop:

```python
from pi_ai import Tool, ToolCall, ToolResultMessage, TextContent
import time

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

context = Context(
    system_prompt="You can check the weather.",
    messages=[UserMessage(content="What's the weather in Tokyo?", timestamp=0)],
    tools=[weather_tool],
)

async def tool_loop():
    while True:
        message = await complete_simple(
            model, context,
            SimpleStreamOptions(api_key="sk-...")
        )
        context.messages.append(message)

        # Check for tool calls
        tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
        if not tool_calls:
            break  # No more tool calls, done

        # Execute tool calls and add results
        for tc in tool_calls:
            result = f"25°C and sunny in {tc.arguments['city']}"
            context.messages.append(ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=[TextContent(text=result)],
                timestamp=int(time.time() * 1000),
            ))

    # Print final response
    for block in message.content:
        if hasattr(block, "text"):
            print(block.text)
```

## Streaming Events

The event stream emits these events in order:

| Event | Description |
|-------|-------------|
| `StartEvent` | Streaming begins |
| `TextStartEvent` | Text block begins |
| `TextDeltaEvent` | Incremental text chunk |
| `TextEndEvent` | Text block complete |
| `ThinkingStartEvent` | Reasoning block begins |
| `ThinkingDeltaEvent` | Reasoning chunk |
| `ThinkingEndEvent` | Reasoning block complete |
| `ToolCallStartEvent` | Tool call begins |
| `ToolCallDeltaEvent` | Tool call argument chunk |
| `ToolCallEndEvent` | Tool call complete |
| `DoneEvent` | Streaming finished successfully |
| `ErrorEvent` | Streaming ended with error |

## Extended Thinking

Enable reasoning for supported models:

```python
message = await complete_simple(
    model, context,
    SimpleStreamOptions(
        api_key="sk-...",
        reasoning="medium",  # minimal, low, medium, high, xhigh
    )
)
```

## License

MIT
