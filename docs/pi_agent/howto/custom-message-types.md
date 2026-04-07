# Custom Message Types

This guide explains how to use custom message types in the agent's conversation transcript, and how to filter them before they reach the LLM.

## Overview

The agent's message list (`agent.state.messages`) can contain any Python object, not just pi-llm message types. This is useful for application-specific messages (status updates, metadata, UI events) that should be part of the transcript but should not be sent to the LLM.

## convert_to_llm

The `convert_to_llm` callback filters messages before each LLM call. The default implementation keeps only `user`, `assistant`, and `toolResult` messages.

```python
def my_convert_to_llm(messages):
    return [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")]
```

## transform_context

The `transform_context` callback provides a second stage of message processing, useful for context window management or message summarization.

## Full example

Content coming soon.

## Next steps

- [Agent Lifecycle](../concepts/agent-lifecycle.md) -- AgentOptions configuration
- [Agent Loop](../concepts/agent-loop.md) -- How messages flow through the loop
