# Getting Started

## Installation

Install both packages:

```bash
pip install pi-ai pi-ai-agent
```

## Set up your API key

pi-python uses OpenAI's API. Export your key:

```bash
export OPENAI_API_KEY=sk-...
```

## Your first LLM call

```python
import asyncio
from pi_ai import get_model, complete_simple, Context, UserMessage
from pi_ai.providers import register_builtin_providers

# Register providers (call once at startup)
register_builtin_providers()

async def main():
    model = get_model("openai", "gpt-4o")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello!", timestamp=0)],
    )
    message = await complete_simple(model, context)
    print(message.content[0].text)

asyncio.run(main())
```

## Your first agent

```python
import asyncio
from pi_agent import Agent, AgentOptions, InitialAgentState, AgentTool, AgentToolResult
from pi_ai import get_model, TextContent, stream_simple
from pi_ai.providers import register_builtin_providers

register_builtin_providers()


class GreetTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="greet",
            label="Greet",
            description="Greet someone by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(text=f"Hello, {params['name']}!")]
        )


async def main():
    agent = Agent(AgentOptions(
        initial_state=InitialAgentState(
            model=get_model("openai", "gpt-4o"),
            system_prompt="You have a greet tool. Use it when asked to greet someone.",
            tools=[GreetTool()],
        ),
        stream_fn=stream_simple,
        get_api_key=lambda _: None,  # uses OPENAI_API_KEY env var
    ))

    # Print events as they happen
    def on_event(event, cancellation):
        if event.type == "message_update":
            if hasattr(event.assistant_message_event, "delta"):
                print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n  [calling {event.tool_name}]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)
    await agent.prompt("Please greet Alice and Bob")

asyncio.run(main())
```

## Next steps

- [pi-ai Quickstart](pi_ai/quickstart.md) — Deeper dive into streaming, events, and tools
- [pi-agent Quickstart](pi_agent/quickstart.md) — Agent lifecycle, hooks, and state management
- [Build a Research Agent](pi_agent/howto/build-research-agent.md) — Full tutorial with web search
