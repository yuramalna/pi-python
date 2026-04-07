# pi-llm-agent

General-purpose agent framework with tool execution and event streaming. Built on [pi-llm](../pi_llm/).

## Features

- **Agent class** — Stateful wrapper managing conversation, tools, and lifecycle
- **Tool execution** — Subclass `AgentTool` or use `from_function()` factory
- **Parallel tools** — Execute tool calls concurrently (default) or sequentially
- **Event streaming** — Subscribe to 10 event types for live UI updates
- **Hooks** — `before_tool_call` / `after_tool_call` for access control and post-processing
- **Steering & follow-up** — Queue messages to interrupt or extend agent runs
- **Cancellation** — Cooperative cancellation via `CancellationToken`

## Installation

```bash
pip install pi-llm-agent
```

Requires `pi-llm` (installed automatically as a dependency).

## Quick Start

```python
import asyncio
from pi_llm_agent import Agent, AgentOptions, InitialAgentState, AgentTool, AgentToolResult
from pi_llm import get_model, TextContent, stream_simple
from pi_llm.providers import register_builtin_providers

register_builtin_providers()


class WeatherTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            label="Get Weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(text=f"25°C and sunny in {params['city']}")]
        )


async def main():
    agent = Agent(AgentOptions(
        initial_state=InitialAgentState(
            model=get_model("openai", "gpt-4o"),
            system_prompt="You are a helpful assistant with weather access.",
            tools=[WeatherTool()],
        ),
        stream_fn=stream_simple,
        get_api_key=lambda _: "sk-...",
    ))

    # Subscribe to events for live output
    agent.subscribe(lambda event, cancel: (
        print(event.type) if event.type != "message_update" else None
    ))

    await agent.prompt("What's the weather in Paris?")

asyncio.run(main())
```

## Event Flow

### Simple prompt (no tools)

```
prompt("Hello")
├─ agent_start
├─ turn_start
├─ message_start    { user message }
├─ message_end      { user message }
├─ message_start    { assistant message }
├─ message_update   { streaming chunks... }
├─ message_end      { assistant message }
├─ turn_end
└─ agent_end
```

### With tool calls

```
prompt("Weather in Tokyo?")
├─ agent_start
├─ turn_start
├─ message_start/end           { user message }
├─ message_start               { assistant + tool call }
├─ message_update...
├─ message_end
├─ tool_execution_start        { get_weather, {city: "Tokyo"} }
├─ tool_execution_end          { result }
├─ message_start/end           { tool result message }
├─ turn_end
│
├─ turn_start                  { next turn }
├─ message_start               { final assistant response }
├─ message_update...
├─ message_end
├─ turn_end
└─ agent_end
```

## Agent Options

```python
agent = Agent(AgentOptions(
    initial_state=InitialAgentState(
        system_prompt="...",
        model=get_model("openai", "gpt-4o"),
        thinking_level="off",           # off, minimal, low, medium, high, xhigh
        tools=[my_tool],
        messages=[],                    # pre-existing history
    ),
    stream_fn=stream_simple,            # custom stream function
    get_api_key=lambda provider: "...", # dynamic API key
    tool_execution="parallel",          # parallel (default) or sequential
    steering_mode="one-at-a-time",      # how steering queue drains
    follow_up_mode="one-at-a-time",     # how follow-up queue drains

    # Hooks
    before_tool_call=my_before_hook,    # block or inspect before execution
    after_tool_call=my_after_hook,      # override results after execution
))
```

## Agent State

```python
agent.state.system_prompt = "New prompt"
agent.state.model = get_model("openai", "gpt-4o-mini")
agent.state.thinking_level = "medium"
agent.state.tools = [new_tool]
agent.state.messages                    # conversation transcript
agent.state.is_streaming                # True during processing
agent.state.streaming_message           # partial message being streamed
agent.state.pending_tool_calls          # set of tool call IDs in progress
```

## Methods

```python
# Prompting
await agent.prompt("Hello")
await agent.prompt("Describe this", images=[image_content])
await agent.continue_()                 # resume from current state

# Control
agent.abort()                           # cancel current run
await agent.wait_for_idle()             # wait for completion
agent.reset()                           # clear everything

# Events
unsubscribe = agent.subscribe(listener) # returns unsubscribe fn
unsubscribe()

# Steering (interrupt between turns)
agent.steer(user_message)
agent.follow_up(user_message)
agent.clear_all_queues()
```

## Hooks

```python
async def my_before_hook(context, cancellation):
    """Block dangerous tools."""
    if context.tool_call.name == "delete_file":
        return BeforeToolCallResult(block=True, reason="Deletion not allowed")
    return None  # allow execution

async def my_after_hook(context, cancellation):
    """Annotate results."""
    return AfterToolCallResult(
        details={**context.result.details, "audited": True}
    )
```

## Low-Level API

For direct control without the Agent class:

```python
from pi_llm_agent import agent_loop, AgentContext, AgentLoopConfig

context = AgentContext(
    system_prompt="You are helpful.",
    messages=[],
    tools=[],
)

config = AgentLoopConfig(
    model=get_model("openai", "gpt-4o"),
    convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "toolResult")],
)

stream = agent_loop([user_message], context, config)
async for event in stream:
    print(event.type)
```

## License

MIT
