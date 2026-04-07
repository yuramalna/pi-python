# pi-llm-agent

General-purpose agent framework with tool execution and event streaming.

Built on [pi-llm](../pi_llm/index.md), pi-llm-agent provides a stateful `Agent` class that manages the multi-turn loop of prompting an LLM, executing tool calls, and repeating until the task is done.

## Key Concepts

- [Agent Lifecycle](concepts/agent-lifecycle.md) — The Agent class, state, and lifecycle methods
- [Agent Loop](concepts/agent-loop.md) — The low-level multi-turn execution engine
- [Events](concepts/events.md) — The 10 agent event types
- [Tools](concepts/tools.md) — Creating tools with `AgentTool`
- [Hooks](concepts/hooks.md) — `before_tool_call` and `after_tool_call`
- [Steering & Follow-up](concepts/steering-followup.md) — Interrupt and extend agent runs
- [Cancellation](concepts/cancellation.md) — Cooperative cancellation

## How-to Guides

- [Build a Research Agent](howto/build-research-agent.md)
- [Custom Message Types](howto/custom-message-types.md)
- [Control Tool Execution](howto/control-tool-execution.md)

## API Reference

See the auto-generated [API Reference](reference/) for complete documentation of all public types and functions.
