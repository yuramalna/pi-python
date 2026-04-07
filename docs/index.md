# pi-python

Unified LLM provider abstraction and agent framework for Python.

## Packages

### [pi-llm](pi_llm/index.md)

Streaming LLM API with model discovery, tool calling, token tracking, and cost calculation.

- Stream or await LLM responses with typed events
- Define tools with JSON Schema, validate arguments automatically
- Track token usage and dollar costs
- Support for extended thinking / reasoning

### [pi-llm-agent](pi_llm_agent/index.md)

Stateful agent framework with tool execution, event streaming, and lifecycle management.

- Multi-turn agent loop: prompt → LLM → tool calls → repeat
- Parallel or sequential tool execution
- Pre/post tool call hooks for access control
- Steering and follow-up message queues
- Cooperative cancellation

## Quick Links

- [Getting Started](getting-started.md) — Install and run your first example
- [pi-llm Quickstart](pi_llm/quickstart.md) — Streaming, completing, and tool calls
- [pi-llm-agent Quickstart](pi_llm_agent/quickstart.md) — Build an agent with tools
- [Build a Research Agent](pi_llm_agent/howto/build-research-agent.md) — End-to-end tutorial
