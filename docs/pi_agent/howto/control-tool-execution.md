# Control Tool Execution

This guide covers how to control whether tools run sequentially or in parallel, and how to use hooks to intercept tool calls.

## Overview

By default, when the LLM requests multiple tool calls in a single response, pi-llm-agent executes them in parallel. You can switch to sequential execution or use hooks to add authorization, logging, or result modification.

## Execution modes

Set the mode in `AgentOptions`:

```python
from pi_llm_agent import AgentOptions

# Parallel (default) -- all tool calls run concurrently
options = AgentOptions(tool_execution="parallel")

# Sequential -- tool calls run one at a time, in order
options = AgentOptions(tool_execution="sequential")
```

## Using hooks

Hooks intercept tool calls before and after execution:

```python
from pi_llm_agent import BeforeToolCallResult

async def before_hook(context, cancellation):
    print(f"About to call: {context.tool_call.name}")
    # Return BeforeToolCallResult(block=True) to prevent execution
    return None

options = AgentOptions(before_tool_call=before_hook)
```

## Full example

Content coming soon.

## Next steps

- [Tools](../concepts/tools.md) -- AgentTool reference
- [Hooks](../concepts/hooks.md) -- Hook types and context objects
- [Events](../concepts/events.md) -- Tool execution events
