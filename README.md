# pi-python

[![CI](https://github.com/yuramalna/pi-python/actions/workflows/ci.yml/badge.svg)](https://github.com/yuramalna/pi-python/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Unified LLM provider abstraction and agent framework for Python.

## Packages

| Package | Description | |
|---------|-------------|---|
| **[pi-ai](packages/pi_ai/)** | Typed streaming interface for LLM API calls with tool calling, extended thinking, and cost tracking | `pip install pi-ai` |
| **[pi-agent](packages/pi_agent/)** | Stateful agent framework with multi-turn tool execution, hooks, and event streaming | `pip install pi-agent` |

## Quick Start

```bash
pip install pi-ai pi-agent
export OPENAI_API_KEY=sk-...
```

### Make an LLM call

```python
import asyncio
from pi_ai import get_model, complete_simple, Context, UserMessage
from pi_ai.providers import register_builtin_providers

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

### Build an agent

```python
import asyncio
from pi_agent import Agent, AgentOptions, InitialAgentState, AgentTool, AgentToolResult
from pi_ai import get_model, TextContent, stream_simple
from pi_ai.providers import register_builtin_providers

register_builtin_providers()

class GreetTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="greet", label="Greet",
            description="Greet someone by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"Hello, {params['name']}!")])

async def main():
    agent = Agent(AgentOptions(
        initial_state=InitialAgentState(
            model=get_model("openai", "gpt-4o"),
            system_prompt="You have a greet tool. Use it when asked to greet someone.",
            tools=[GreetTool()],
        ),
        stream_fn=stream_simple,
        get_api_key=lambda _: None,
    ))

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

## Features

**pi-ai** - LLM Provider Abstraction
- Streaming and non-streaming APIs (`stream_simple`, `complete_simple`)
- 12 typed streaming event types with async iteration
- Tool calling with JSON Schema validation
- Extended thinking / reasoning support (5 levels)
- Token usage and cost tracking (42 OpenAI models)
- Context overflow detection across 20+ providers
- Prompt caching support

**pi-agent** - Agent Framework
- Stateful `Agent` class with multi-turn conversation
- Tool execution (sequential or parallel) with `AgentTool`
- 10 agent event types with subscriber pattern
- Before/after tool call hooks
- Steering and follow-up message queues
- Cooperative cancellation
- Low-level `agent_loop` for custom control flows

## Documentation

Full documentation is available at **[yuramalna.github.io/pi-python](https://yuramalna.github.io/pi-python/)**.

- [Getting Started](https://yuramalna.github.io/pi-python/getting-started/)
- [pi-ai Concepts](https://yuramalna.github.io/pi-python/pi_ai/)
- [pi-agent Concepts](https://yuramalna.github.io/pi-python/pi_agent/)
- [API Reference](https://yuramalna.github.io/pi-python/pi_ai/reference/)

## Development

```bash
git clone https://github.com/yuramalna/pi-python.git
cd pi-python
python -m venv .venv && source .venv/bin/activate

# Install both packages in dev mode
pip install -e "packages/pi_ai[dev]"
pip install -e "packages/pi_agent[dev]"

# Install docs dependencies
pip install mkdocs-material 'mkdocstrings[python]' mkdocs-gen-files mkdocs-section-index

# Run tests
pytest packages/pi_ai/tests/ --ignore=packages/pi_ai/tests/integration
pytest packages/pi_agent/tests/ --ignore=packages/pi_agent/tests/integration

# Serve docs locally
mkdocs serve
```

## License

[MIT](LICENSE)
