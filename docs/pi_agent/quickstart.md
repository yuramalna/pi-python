# pi-llm-agent Quickstart

This guide walks you through building your first agent: creating tools, configuring the agent, subscribing to events, and sending prompts.

## Installation

```bash
pip install pi-llm-agent
```

pi-llm-agent depends on pi-llm, which will be installed automatically.

## Your first agent

An `Agent` manages the multi-turn loop: it sends a prompt to the LLM, executes any tool calls, feeds the results back, and repeats until the task is done.

```python
import asyncio
import os

from pi_agent import (
    Agent,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    InitialAgentState,
)
from pi_ai import TextContent, get_model, stream_simple
from pi_ai.providers import register_builtin_providers

register_builtin_providers()


# 1. Define a tool
class GreetTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="greet",
            label="Greet",
            description="Generate a greeting for a person",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person's name"},
                },
                "required": ["name"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        name = params["name"]
        return AgentToolResult(
            content=[TextContent(text=f"Hello, {name}! Welcome.")]
        )


# 2. Create the agent
agent = Agent(AgentOptions(
    initial_state=InitialAgentState(
        model=get_model("openai", "gpt-4o"),
        system_prompt="You are a helpful assistant. Use the greet tool when asked to greet someone.",
        tools=[GreetTool()],
    ),
    get_api_key=lambda provider: os.environ.get("OPENAI_API_KEY"),
))


# 3. Subscribe to events
def on_event(event, cancellation):
    match event.type:
        case "message_update":
            # Incremental text from the LLM
            inner = event.assistant_message_event
            if hasattr(inner, "delta") and inner.type == "text_delta":
                print(inner.delta, end="", flush=True)
        case "tool_execution_start":
            print(f"\n[Calling tool: {event.tool_name}]")
        case "tool_execution_end":
            print(f"[Tool result: {event.result.content[0].text}]")
        case "agent_end":
            print("\n[Agent finished]")


agent.subscribe(on_event)


# 4. Send a prompt
async def main():
    await agent.prompt("Please greet Alice and Bob.")


asyncio.run(main())
```

## How it works

1. **`agent.prompt("...")`** adds a `UserMessage` to the transcript, then runs the agent loop.
2. The agent loop calls the LLM via `stream_simple`, streaming events.
3. When the LLM emits tool calls, the agent executes them (in parallel by default).
4. Tool results are appended to the transcript and the LLM is called again.
5. This repeats until the LLM responds with text only (no tool calls).
6. Events are emitted at every stage so subscribers can react in real time.

## Reading agent state

The `agent.state` object gives you access to the current conversation:

```python
# After prompt completes:
print(f"Total messages: {len(agent.state.messages)}")
print(f"Is streaming: {agent.state.is_streaming}")
print(f"Model: {agent.state.model.id}")
```

## Using from_function for simple tools

For tools that do not need a class, use the factory method:

```python
from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent


async def reverse_text(tool_call_id, params, cancellation=None, on_update=None):
    text = params["text"]
    return AgentToolResult(
        content=[TextContent(text=text[::-1])]
    )


reverse_tool = AgentTool.from_function(
    name="reverse_text",
    label="Reverse Text",
    description="Reverse a string of text",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to reverse"},
        },
        "required": ["text"],
    },
    fn=reverse_text,
)
```

## Cancellation

Cancel a running agent with `agent.abort()`:

```python
import asyncio

async def main():
    # Start the prompt in a task
    task = asyncio.create_task(agent.prompt("Write a very long essay."))

    # Cancel after 2 seconds
    await asyncio.sleep(2)
    agent.abort()

    await task  # The prompt finishes with an "aborted" stop reason
```

## Steering and follow-up

Inject messages into the conversation while the agent is running:

```python
# Steering: injected between turns (before the next LLM call)
agent.steer(UserMessage(
    content="Focus on the technical aspects.",
    timestamp=int(time.time() * 1000),
))

# Follow-up: processed after the current run completes
agent.follow_up(UserMessage(
    content="Now summarize what you found.",
    timestamp=int(time.time() * 1000),
))
```

## Next steps

- [Agent Lifecycle](concepts/agent-lifecycle.md) -- The Agent class, state, and lifecycle
- [Events](concepts/events.md) -- The 10 agent event types
- [Tools](concepts/tools.md) -- Creating tools with AgentTool
- [Build a Research Agent](howto/build-research-agent.md) -- End-to-end tutorial
