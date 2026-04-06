# Events

When you stream a response with `stream_simple` or `stream`, pi-ai emits a sequence of typed events. There are 12 event types in total, grouped into four families: lifecycle, text, thinking, and tool calls.

## Event type overview

| Event | Type string | Description |
|---|---|---|
| `StartEvent` | `"start"` | Streaming has begun |
| `TextStartEvent` | `"text_start"` | A new text block started |
| `TextDeltaEvent` | `"text_delta"` | An incremental chunk of text |
| `TextEndEvent` | `"text_end"` | A text block completed |
| `ThinkingStartEvent` | `"thinking_start"` | A reasoning block started |
| `ThinkingDeltaEvent` | `"thinking_delta"` | An incremental chunk of reasoning |
| `ThinkingEndEvent` | `"thinking_end"` | A reasoning block completed |
| `ToolCallStartEvent` | `"toolcall_start"` | A tool call started |
| `ToolCallDeltaEvent` | `"toolcall_delta"` | Incremental tool call arguments |
| `ToolCallEndEvent` | `"toolcall_end"` | A tool call completed |
| `DoneEvent` | `"done"` | Streaming completed successfully |
| `ErrorEvent` | `"error"` | Streaming ended with an error |

## Common fields

Every event has a `type` field (a string) and most events carry a `partial` field -- the `AssistantMessage` being built incrementally. The `partial` message is updated in place as events arrive, so by the time you receive `DoneEvent`, `partial` contains the complete message.

Events that belong to a content block also have a `content_index` indicating which block within `partial.content` the event pertains to.

## Lifecycle events

### StartEvent

Emitted once at the beginning of every stream.

```python
@dataclass
class StartEvent:
    partial: AssistantMessage
    type: str = "start"
```

### DoneEvent

Emitted when streaming completes successfully.

```python
@dataclass
class DoneEvent:
    reason: StopReason  # "stop", "length", "toolUse", etc.
    message: AssistantMessage  # The complete message
    type: str = "done"
```

### ErrorEvent

Emitted when streaming ends due to an error or cancellation.

```python
@dataclass
class ErrorEvent:
    reason: StopReason  # "error" or "aborted"
    error: AssistantMessage  # Contains error_message
    type: str = "error"
```

## Text events

Text events track the incremental construction of a `TextContent` block.

```python
from pi_ai import TextDeltaEvent, TextEndEvent

async for event in event_stream:
    if isinstance(event, TextDeltaEvent):
        # event.delta is a string chunk
        print(event.delta, end="", flush=True)
    elif isinstance(event, TextEndEvent):
        # event.content is the full text of this block
        print(f"\n[Complete text: {len(event.content)} chars]")
```

### TextStartEvent

Signals the start of a new text content block. The `content_index` tells you which position in `partial.content` will hold the completed `TextContent`.

### TextDeltaEvent

Contains an incremental `delta` string. Concatenating all deltas for a given `content_index` produces the full text.

### TextEndEvent

Contains the complete `content` string for this block.

## Thinking events

Thinking events mirror text events but track `ThinkingContent` blocks. They only appear when reasoning is enabled.

```python
from pi_ai import ThinkingDeltaEvent, SimpleStreamOptions

options = SimpleStreamOptions(reasoning="medium")
event_stream = stream_simple(model, context, options)

async for event in event_stream:
    if isinstance(event, ThinkingDeltaEvent):
        print(f"[thinking] {event.delta}", end="")
```

### ThinkingStartEvent / ThinkingDeltaEvent / ThinkingEndEvent

Same structure as the text family: `content_index`, `delta` (for the delta event), and `content` (for the end event). The `ThinkingContent` block may have a `thinking_signature` used for cross-provider handoff.

## Tool call events

Tool call events track the incremental construction of a `ToolCall` block. The arguments are streamed as JSON string chunks.

```python
from pi_ai import ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent

async for event in event_stream:
    if isinstance(event, ToolCallStartEvent):
        print(f"[Tool call starting at index {event.content_index}]")
    elif isinstance(event, ToolCallDeltaEvent):
        # event.delta is a chunk of the JSON arguments string
        pass
    elif isinstance(event, ToolCallEndEvent):
        # event.tool_call is the complete ToolCall with parsed arguments
        tc = event.tool_call
        print(f"[Tool: {tc.name}({tc.arguments})]")
```

### ToolCallStartEvent

Signals the start of a new tool call block.

### ToolCallDeltaEvent

Contains a `delta` string -- an incremental chunk of the JSON-serialized arguments.

### ToolCallEndEvent

Contains the fully parsed `ToolCall` object with `id`, `name`, and `arguments` (a dict).

## Event sequence

A typical text-only response produces:

```
StartEvent -> TextStartEvent -> TextDeltaEvent* -> TextEndEvent -> DoneEvent
```

A response with thinking enabled:

```
StartEvent -> ThinkingStartEvent -> ThinkingDeltaEvent* -> ThinkingEndEvent
           -> TextStartEvent -> TextDeltaEvent* -> TextEndEvent -> DoneEvent
```

A response that invokes a tool:

```
StartEvent -> ToolCallStartEvent -> ToolCallDeltaEvent* -> ToolCallEndEvent -> DoneEvent
```

Multiple content blocks can appear in a single response. For example, the model might emit text, then a tool call, or multiple tool calls in sequence.

## The AssistantMessageEvent union

All 12 event types are members of the `AssistantMessageEvent` union:

```python
from pi_ai import AssistantMessageEvent

# Type-safe matching with isinstance:
async for event in event_stream:
    if isinstance(event, TextDeltaEvent):
        ...
    elif isinstance(event, DoneEvent):
        ...

# Or match on the type string:
async for event in event_stream:
    match event.type:
        case "text_delta":
            ...
        case "done":
            ...
```

## Next steps

- [Streaming](streaming.md) -- The streaming functions and EventStream
- [Tools](tools.md) -- Defining tools for tool call events
- [Messages & Context](messages-and-context.md) -- The AssistantMessage that events build
