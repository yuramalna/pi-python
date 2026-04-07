# Events

The agent emits events at every stage of processing. There are 10 event types, grouped into four families: agent lifecycle, turn lifecycle, message streaming, and tool execution.

## Event type overview

| Event | Type string | Description |
|---|---|---|
| `AgentStartEvent` | `"agent_start"` | Agent loop has begun |
| `AgentEndEvent` | `"agent_end"` | Agent loop has finished |
| `TurnStartEvent` | `"turn_start"` | A new turn is starting |
| `TurnEndEvent` | `"turn_end"` | A turn has completed |
| `AgentMessageStartEvent` | `"message_start"` | A message is being added |
| `AgentMessageUpdateEvent` | `"message_update"` | Incremental streaming update |
| `AgentMessageEndEvent` | `"message_end"` | A message is complete |
| `ToolExecutionStartEvent` | `"tool_execution_start"` | A tool call is starting |
| `ToolExecutionUpdateEvent` | `"tool_execution_update"` | A tool is streaming partial results |
| `ToolExecutionEndEvent` | `"tool_execution_end"` | A tool call has completed |

## Subscribing to events

Use `agent.subscribe()` to receive events. The callback receives the event and a `CancellationToken`:

```python
from pi_agent import CancellationToken

def on_event(event, cancellation: CancellationToken):
    print(f"Event: {event.type}")

unsubscribe = agent.subscribe(on_event)
```

The callback can be sync or async:

```python
async def on_event_async(event, cancellation):
    await log_event(event)

unsubscribe = agent.subscribe(on_event_async)
```

Call the returned function to unsubscribe:

```python
unsubscribe()
```

## Agent lifecycle events

### AgentStartEvent

Emitted once when the agent loop begins processing (at the start of `prompt()` or `continue_()`).

```python
from pi_agent import AgentStartEvent

if isinstance(event, AgentStartEvent):
    print("Agent started")
```

### AgentEndEvent

Emitted when the agent loop finishes. Contains all messages added during the run.

```python
from pi_agent import AgentEndEvent

if isinstance(event, AgentEndEvent):
    print(f"Agent finished. {len(event.messages)} new messages.")
```

## Turn lifecycle events

A "turn" is one LLM call plus any tool executions triggered by the response. An agent run consists of one or more turns.

### TurnStartEvent

Emitted at the start of each turn.

### TurnEndEvent

Emitted at the end of a turn. Contains the assistant message and any tool results.

```python
from pi_agent import TurnEndEvent

if isinstance(event, TurnEndEvent):
    msg = event.message
    print(f"Turn complete. Tool results: {len(event.tool_results)}")
```

## Message streaming events

These events track the lifecycle of messages being added to the conversation.

### AgentMessageStartEvent

Emitted when a message begins (user messages, assistant messages being streamed, tool results).

```python
from pi_agent import AgentMessageStartEvent

if isinstance(event, AgentMessageStartEvent):
    msg = event.message
    if hasattr(msg, "role"):
        print(f"Message starting: {msg.role}")
```

### AgentMessageUpdateEvent

Emitted for incremental streaming updates on assistant messages. Contains both the partial agent message and the underlying pi-llm streaming event.

```python
from pi_agent import AgentMessageUpdateEvent
from pi_ai import TextDeltaEvent

if isinstance(event, AgentMessageUpdateEvent):
    inner = event.assistant_message_event
    if isinstance(inner, TextDeltaEvent):
        print(inner.delta, end="", flush=True)
```

This is the primary event for building real-time UIs -- the `assistant_message_event` gives you access to the full pi-llm event (text deltas, thinking deltas, tool call deltas, etc.).

### AgentMessageEndEvent

Emitted when a message is complete and has been added to the transcript.

```python
from pi_agent import AgentMessageEndEvent

if isinstance(event, AgentMessageEndEvent):
    msg = event.message
    print(f"Message complete: {msg.role if hasattr(msg, 'role') else 'custom'}")
```

## Tool execution events

These events bracket individual tool executions within a turn.

### ToolExecutionStartEvent

Emitted when a tool call begins execution.

```python
from pi_agent import ToolExecutionStartEvent

if isinstance(event, ToolExecutionStartEvent):
    print(f"Executing: {event.tool_name}(id={event.tool_call_id})")
    print(f"Arguments: {event.args}")
```

### ToolExecutionUpdateEvent

Emitted when a tool streams a partial result via its `on_update` callback. Useful for long-running tools that report progress.

```python
from pi_agent import ToolExecutionUpdateEvent

if isinstance(event, ToolExecutionUpdateEvent):
    print(f"[{event.tool_name}] progress: {event.partial_result}")
```

### ToolExecutionEndEvent

Emitted when a tool call completes.

```python
from pi_agent import ToolExecutionEndEvent

if isinstance(event, ToolExecutionEndEvent):
    status = "error" if event.is_error else "success"
    print(f"[{event.tool_name}] {status}")
```

## Event sequence diagrams

### Simple text response (no tool calls)

```
AgentStartEvent
  TurnStartEvent
    AgentMessageStartEvent  (user message)
    AgentMessageEndEvent
    AgentMessageStartEvent  (assistant message, streaming begins)
    AgentMessageUpdateEvent (text_delta) ...
    AgentMessageEndEvent
  TurnEndEvent
AgentEndEvent
```

### Response with tool calls

```
AgentStartEvent
  TurnStartEvent
    AgentMessageStartEvent  (user message)
    AgentMessageEndEvent
    AgentMessageStartEvent  (assistant message with tool calls)
    AgentMessageUpdateEvent (toolcall_delta) ...
    AgentMessageEndEvent
    ToolExecutionStartEvent (tool A)
    ToolExecutionStartEvent (tool B)    -- parallel by default
    ToolExecutionEndEvent   (tool A)
    ToolExecutionEndEvent   (tool B)
    AgentMessageStartEvent  (tool result A)
    AgentMessageEndEvent
    AgentMessageStartEvent  (tool result B)
    AgentMessageEndEvent
  TurnEndEvent
  TurnStartEvent
    AgentMessageStartEvent  (assistant message, final text)
    AgentMessageUpdateEvent (text_delta) ...
    AgentMessageEndEvent
  TurnEndEvent
AgentEndEvent
```

## Building an event printer

A common pattern is a function that prints events for debugging:

```python
from pi_agent import (
    AgentStartEvent,
    AgentEndEvent,
    AgentMessageUpdateEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
)
from pi_ai import TextDeltaEvent


def make_event_printer():
    """Create an event subscriber that prints agent activity."""
    last_text_len = 0

    def on_event(event, cancellation):
        nonlocal last_text_len

        if isinstance(event, AgentStartEvent):
            print("--- Agent started ---")

        elif isinstance(event, AgentMessageUpdateEvent):
            inner = event.assistant_message_event
            if isinstance(inner, TextDeltaEvent):
                print(inner.delta, end="", flush=True)

        elif isinstance(event, ToolExecutionStartEvent):
            print(f"\n[Tool: {event.tool_name}({event.args})]")

        elif isinstance(event, ToolExecutionEndEvent):
            status = "error" if event.is_error else "done"
            print(f"[Tool {event.tool_name}: {status}]")

        elif isinstance(event, AgentEndEvent):
            print("\n--- Agent finished ---")

    return on_event


# Usage
agent.subscribe(make_event_printer())
```

## The AgentEvent union

All 10 event types are members of the `AgentEvent` union type:

```python
from pi_agent import AgentEvent
```

You can match on the `type` string or use `isinstance` checks.

## Next steps

- [Agent Lifecycle](agent-lifecycle.md) -- The Agent class and state management
- [Tools](tools.md) -- Creating tools that generate tool execution events
- [Hooks](hooks.md) -- Intercepting tool calls with before/after hooks
