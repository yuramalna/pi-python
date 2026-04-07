# Streaming

pi-llm provides four top-level functions for making LLM calls. Two are "simple" (handle reasoning automatically) and two are "raw" (accept provider-specific options). Each pair has a streaming variant and a non-streaming convenience wrapper.

## The four functions

| Function | Returns | Use case |
|---|---|---|
| `stream_simple(model, context, options)` | `AssistantMessageEventStream` | Recommended for most use cases |
| `complete_simple(model, context, options)` | `AssistantMessage` | When you only need the final result |
| `stream(model, context, options)` | `AssistantMessageEventStream` | When you need provider-specific options |
| `complete(model, context, options)` | `AssistantMessage` | Non-streaming with provider-specific options |

### stream_simple

The recommended entry point. Accepts `SimpleStreamOptions` which includes reasoning level support.

```python
from pi_ai import Context, SimpleStreamOptions, get_model, stream_simple
from pi_ai.providers import register_builtin_providers

register_builtin_providers()

model = get_model("openai", "gpt-4o")
context = Context(system_prompt="You are helpful.", messages=[])
options = SimpleStreamOptions(reasoning="medium")

event_stream = stream_simple(model, context, options)
```

### complete_simple

A convenience wrapper that internally calls `stream_simple` and awaits the final result:

```python
from pi_ai import Context, SimpleStreamOptions, complete_simple, get_model

message = await complete_simple(model, context, SimpleStreamOptions())
print(message.content[0].text)
```

### stream / complete

The lower-level pair. These accept `StreamOptions` (without reasoning level) and are useful when you need to pass provider-specific options like `cache_retention` or `session_id` without the reasoning abstraction.

```python
from pi_ai import Context, StreamOptions, get_model, stream

options = StreamOptions(
    temperature=0.7,
    max_tokens=1024,
    cache_retention="long",
)

event_stream = stream(model, context, options)
```

## EventStream

All streaming functions return an `AssistantMessageEventStream`, which is a specialization of the generic `EventStream[T, R]`.

### Async iteration

Consume events as they arrive:

```python
from pi_ai import TextDeltaEvent, DoneEvent, ErrorEvent

event_stream = stream_simple(model, context, SimpleStreamOptions())

async for event in event_stream:
    match event.type:
        case "text_delta":
            print(event.delta, end="", flush=True)
        case "done":
            print(f"\nFinished: {event.reason}")
        case "error":
            print(f"\nError: {event.error.error_message}")
```

### Getting the final result

Use `.result()` to await the completed `AssistantMessage`. This works whether or not you consumed the events:

```python
event_stream = stream_simple(model, context, SimpleStreamOptions())
message = await event_stream.result()

# message is an AssistantMessage with all content blocks
for block in message.content:
    if block.type == "text":
        print(block.text)
```

### Producer API

The `EventStream` is push-based. Providers use `push()` and `end()` to emit events:

```python
from pi_ai import AssistantMessageEventStream

# This is what providers do internally:
event_stream = AssistantMessageEventStream()
event_stream.push(start_event)
event_stream.push(text_delta_event)
# ...
event_stream.push(done_event)  # Triggers result resolution
event_stream.end()             # Signals no more events
```

The `push()` method detects terminal events (`DoneEvent` or `ErrorEvent`) and resolves the result future automatically. After `end()`, further `push()` calls are silently dropped.

## SimpleStreamOptions

Options for the simple API:

| Field | Type | Default | Description |
|---|---|---|---|
| `reasoning` | `ThinkingLevel \| None` | `None` | Extended thinking level |
| `thinking_budgets` | `dict[str, int] \| None` | `None` | Custom token budgets per level |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens |
| `api_key` | `str \| None` | `None` | API key override |
| `cancel_event` | `asyncio.Event \| None` | `None` | Cancellation signal |
| `cache_retention` | `CacheRetention \| None` | `None` | Prompt caching preference |
| `session_id` | `str \| None` | `None` | Session ID for caching |
| `headers` | `dict[str, str] \| None` | `None` | Additional HTTP headers |
| `metadata` | `dict[str, Any] \| None` | `None` | Arbitrary metadata for the provider |

## StreamOptions

The base options class (used by `stream` / `complete`). Identical to `SimpleStreamOptions` but without `reasoning` or `thinking_budgets`.

## Cancellation

Pass an `asyncio.Event` to cancel a streaming request:

```python
import asyncio

cancel = asyncio.Event()
options = SimpleStreamOptions(cancel_event=cancel)
event_stream = stream_simple(model, context, options)

# In another coroutine:
cancel.set()  # Cancels the in-flight request
```

## Next steps

- [Events](events.md) -- The 12 event types emitted during streaming
- [Tools](tools.md) -- Adding tools to the context
- [Thinking / Reasoning](thinking.md) -- Reasoning levels and thinking events
