# Agent Loop

The agent loop is the low-level multi-turn execution engine that powers the `Agent` class in pi-llm-agent. It manages the cycle of calling the LLM, executing tool calls, and repeating until the task is complete.

## Overview

The loop operates at a lower level than the `Agent` class. While `Agent` provides state management, event dispatch, and message queueing, the agent loop focuses purely on the turn-by-turn execution logic.

## Key functions

- **`run_agent_loop(messages, context, config, event_sink, cancellation, stream_fn)`** -- Run the loop with new input messages.
- **`run_agent_loop_continue(context, config, event_sink, cancellation, stream_fn)`** -- Continue from existing context.
- **`agent_loop()`** / **`agent_loop_continue()`** -- Lower-level async generators.

## Turn structure

A single turn consists of:

1. Build context from system prompt, messages, and tools.
2. Call the LLM via the `stream_fn`.
3. Consume the event stream, building the assistant message.
4. If the message contains tool calls, execute them and append results.
5. Poll for steering messages.
6. If there are pending tool results or steering messages, start a new turn.

## AgentLoopConfig

Content coming soon.

## Next steps

- [Agent Lifecycle](agent-lifecycle.md) -- The Agent class that wraps the loop
- [Events](events.md) -- Events emitted during the loop
- [Steering & Follow-up](steering-followup.md) -- Message injection between turns
