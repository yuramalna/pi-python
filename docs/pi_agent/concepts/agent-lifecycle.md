# Agent Lifecycle

The `Agent` class is the primary public API for pi-llm-agent. It wraps the low-level agent loop, managing conversation state, tool execution, event dispatch, and message queueing.

## Creating an agent

An agent is created with `AgentOptions`, which configures its initial state and behavior:

```python
from pi_agent import Agent, AgentOptions, InitialAgentState
from pi_ai import get_model

agent = Agent(AgentOptions(
    initial_state=InitialAgentState(
        model=get_model("openai", "gpt-4o"),
        system_prompt="You are a helpful assistant.",
        tools=[my_tool],
        thinking_level="off",
    ),
    get_api_key=lambda provider: os.environ.get("OPENAI_API_KEY"),
))
```

### InitialAgentState

Sets the starting conditions:

| Field | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `""` | System instructions for the LLM |
| `model` | `Model \| None` | `None` | LLM model to use |
| `thinking_level` | `AgentThinkingLevel` | `"off"` | Reasoning level (`"off"`, `"minimal"`, ..., `"xhigh"`) |
| `tools` | `list[AgentTool] \| None` | `None` | Available tools |
| `messages` | `list[Any] \| None` | `None` | Pre-existing conversation history |

### AgentOptions

Configures agent behavior:

| Field | Type | Default | Description |
|---|---|---|---|
| `initial_state` | `InitialAgentState \| None` | `None` | Starting state |
| `convert_to_llm` | `Callable` | `None` | Custom message filter |
| `transform_context` | `Callable` | `None` | Pre-process messages before LLM call |
| `stream_fn` | `StreamFn` | `None` | Custom streaming function |
| `get_api_key` | `Callable` | `None` | Dynamic API key resolver |
| `before_tool_call` | `BeforeToolCallHook` | `None` | Pre-execution hook |
| `after_tool_call` | `AfterToolCallHook` | `None` | Post-execution hook |
| `steering_mode` | `QueueMode` | `"one-at-a-time"` | How steering messages drain |
| `follow_up_mode` | `QueueMode` | `"one-at-a-time"` | How follow-up messages drain |
| `tool_execution` | `ToolExecutionMode` | `"parallel"` | Sequential or parallel tool execution |

## Agent state

Access the agent's mutable state via `agent.state`:

```python
state = agent.state

# Conversation data
state.system_prompt      # str
state.model              # Model
state.thinking_level     # AgentThinkingLevel
state.tools              # list[AgentTool]
state.messages           # list[Any]

# Runtime state (read-only, managed by the agent)
state.is_streaming       # bool -- True while processing
state.streaming_message  # The partial message being built, or None
state.pending_tool_calls # set[str] -- Tool call IDs in progress
state.error_message      # str | None -- Last error
```

You can modify `system_prompt`, `model`, `thinking_level`, `tools`, and `messages` between agent runs. Changes to `tools` and `messages` are copy-on-assign to prevent accidental mutation.

## Lifecycle methods

### prompt()

Send a prompt to the agent. Accepts a string, a message object, or a list of messages:

```python
# String prompt (automatically wrapped in UserMessage)
await agent.prompt("Hello!")

# With images
from pi_ai import ImageContent
await agent.prompt("Describe this image.", images=[image_content])

# Raw message or list of messages
await agent.prompt(user_message)
await agent.prompt([msg1, msg2])
```

Raises `RuntimeError` if the agent is already processing.

### continue_()

Resume processing from the current conversation state without adding a new message:

```python
await agent.continue_()
```

The last message must be `user` or `toolResult` (not `assistant`). If the last message is `assistant`, the method drains steering or follow-up queues instead.

### abort()

Cancel the current processing run:

```python
agent.abort()
```

Uses cooperative cancellation via `CancellationToken`. The agent finishes with an `"aborted"` stop reason.

### wait_for_idle()

Wait until the agent finishes processing:

```python
await agent.wait_for_idle()
```

### reset()

Clear all messages, state, and queues:

```python
agent.reset()
```

## Event subscription

Subscribe to agent events with a sync or async callback:

```python
def on_event(event, cancellation):
    print(event.type)

unsubscribe = agent.subscribe(on_event)

# Later, stop receiving events:
unsubscribe()
```

The listener receives `(AgentEvent, CancellationToken)`. See [Events](events.md) for the full event catalog.

## Message queueing

The agent provides two message queues for injecting messages into the conversation:

### Steering

Steering messages are injected between turns (after the LLM responds, before the next call):

```python
agent.steer(message)
```

### Follow-up

Follow-up messages are processed after the current agent run completes:

```python
agent.follow_up(message)
```

### Queue modes

Both queues support two drain modes:

- `"one-at-a-time"` (default) -- Process one queued message per turn.
- `"all"` -- Process all queued messages at once.

```python
agent.steering_mode = "all"
agent.follow_up_mode = "one-at-a-time"
```

### Queue management

```python
agent.clear_steering_queue()
agent.clear_follow_up_queue()
agent.clear_all_queues()
agent.has_queued_messages()  # bool
```

## Lifecycle flow

```
Agent created
    |
    v
prompt("Hello!") ---------> RuntimeError if already processing
    |
    v
[idle_event cleared]
[CancellationToken created]
[state.is_streaming = True]
    |
    v
run_agent_loop()
    |-- Turn 1: LLM call -> tool execution -> results
    |-- [steering queue polled]
    |-- Turn 2: LLM call -> text response
    |-- [follow-up queue polled]
    |-- (continue if follow-ups...)
    |
    v
[AgentEndEvent emitted]
[state.is_streaming = False]
[CancellationToken = None]
[idle_event set]
```

## Next steps

- [Agent Loop](agent-loop.md) -- The low-level multi-turn engine
- [Events](events.md) -- The 10 agent event types
- [Tools](tools.md) -- Creating tools with AgentTool
- [Steering & Follow-up](steering-followup.md) -- Queue modes and injection
