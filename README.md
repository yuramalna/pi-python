<p align="center">
  <h1 align="center">pi-python</h1>
  <p align="center">
    Unified LLM provider abstraction and agent framework for Python
  </p>
</p>

<p align="center">
  <a href="https://github.com/yuramalna/pi-python/actions/workflows/ci.yml"><img src="https://github.com/yuramalna/pi-python/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://yuramalna.github.io/pi-python/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
</p>

---

**pi-python** is a Python port of the LLM and agent packages from [pi-mono](https://github.com/badlogic/pi-mono) by [Mario Zechner](https://github.com/badlogic). It provides a clean, typed interface for building LLM-powered applications with streaming, tool calling, and autonomous agents.

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| **[pi-llm](packages/pi_ai/)** | Typed streaming interface for LLM calls &mdash; tool calling, extended thinking, cost tracking | `pip install pi-llm` |
| **[pi-llm-agent](packages/pi_agent/)** | Stateful agent framework &mdash; multi-turn tool execution, hooks, event streaming | `pip install pi-llm-agent` |

## Quick Start

```bash
pip install pi-llm pi-llm-agent
export OPENAI_API_KEY=sk-...
```

### Make an LLM call

```python
import asyncio
from pi_llm import get_model, complete_simple, Context, UserMessage
from pi_llm.providers import register_builtin_providers

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

### Build an agent with tools

```python
import asyncio
from pi_llm_agent import Agent, AgentOptions, InitialAgentState, AgentTool, AgentToolResult
from pi_llm import get_model, TextContent, stream_simple
from pi_llm.providers import register_builtin_providers

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
        if event.type == "message_update" and hasattr(event.assistant_message_event, "delta"):
            print(event.assistant_message_event.delta, end="", flush=True)
        elif event.type == "tool_execution_start":
            print(f"\n  [calling {event.tool_name}]")
        elif event.type == "agent_end":
            print()

    agent.subscribe(on_event)
    await agent.prompt("Please greet Alice and Bob")

asyncio.run(main())
```

## Why pi-python?

Most Python LLM libraries either lock you into one provider or bury you in abstraction layers. pi-python gives you:

- **Typed streaming events** &mdash; 12 event types you can `async for` over, not string parsing
- **Real tool execution** &mdash; JSON Schema validation, before/after hooks, sequential or parallel
- **Agent state you control** &mdash; inspect messages, pending tool calls, and streaming state at any point
- **Extended thinking** &mdash; 5 reasoning levels from `minimal` to `xhigh`
- **Cost awareness** &mdash; token usage and dollar cost tracking for 42 OpenAI models

## Features

<table>
<tr>
<td width="50%" valign="top">

### pi-llm &mdash; LLM Abstraction

- `stream_simple` / `complete_simple` &mdash; streaming and non-streaming
- 12 typed streaming event types with async iteration
- Tool calling with JSON Schema validation
- Extended thinking / reasoning (5 levels)
- Token usage and cost tracking
- Context overflow detection (20+ providers)
- Prompt caching support
- Pydantic models throughout

</td>
<td width="50%" valign="top">

### pi-llm-agent &mdash; Agent Framework

- Stateful `Agent` class with multi-turn conversation
- `AgentTool` with `execute()` &mdash; sequential or parallel
- 10 agent event types with subscriber pattern
- Before/after tool call hooks
- Steering and follow-up message queues
- Cooperative cancellation with `CancellationToken`
- Low-level `agent_loop` for custom control flows
- Faux provider for deterministic testing

</td>
</tr>
</table>

## Documentation

Full documentation at **[yuramalna.github.io/pi-python](https://yuramalna.github.io/pi-python/)**

| | |
|---|---|
| [Getting Started](https://yuramalna.github.io/pi-python/getting-started/) | Install, first LLM call, first agent |
| [pi-ai Concepts](https://yuramalna.github.io/pi-python/pi_ai/) | Streaming, events, tools, thinking, cost tracking |
| [pi-agent Concepts](https://yuramalna.github.io/pi-python/pi_agent/) | Agent lifecycle, loop, hooks, steering, cancellation |
| [API Reference](https://yuramalna.github.io/pi-python/pi_ai/reference/) | Auto-generated from source docstrings |

## Development

```bash
git clone https://github.com/yuramalna/pi-python.git
cd pi-python
python -m venv .venv && source .venv/bin/activate

# Install both packages in dev mode
pip install -e "packages/pi_ai[dev]"
pip install -e "packages/pi_agent[dev]"  # PyPI name: pi-llm-agent

# Install docs dependencies
pip install mkdocs-material 'mkdocstrings[python]' mkdocs-gen-files mkdocs-section-index

# Run tests (322 total)
pytest packages/pi_ai/tests/ --ignore=packages/pi_ai/tests/integration
pytest packages/pi_agent/tests/ --ignore=packages/pi_agent/tests/integration

# Lint
ruff check packages/pi_ai/src/ packages/pi_agent/src/

# Serve docs locally
mkdocs serve
```

## Acknowledgments

This project is a Python port of the `@anthropic/pi-ai` and `@anthropic/pi-agent` TypeScript packages from [**pi-mono**](https://github.com/badlogic/pi-mono) by [**Mario Zechner**](https://github.com/badlogic) ([@badaboroc](https://twitter.com/badaboroc)). The original pi-mono is a comprehensive AI agent toolkit featuring a coding agent CLI, unified LLM API, TUI & web UI libraries, and more. I'm grateful for his excellent architecture and open-source work that made this port possible.

## License

[MIT](LICENSE)
