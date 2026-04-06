# Hooks

Hooks let you intercept tool calls before and after execution. Use them for authorization, logging, argument transformation, or result modification.

## Overview

pi-agent provides two hook points:

- **`before_tool_call`** -- Called before a tool executes. Can block execution.
- **`after_tool_call`** -- Called after a tool executes. Can override the result.

Both hooks are async functions configured via `AgentOptions`.

## BeforeToolCallHook

The `before_tool_call` hook receives a `BeforeToolCallContext` and an optional `CancellationToken`. It returns a `BeforeToolCallResult` (or `None` to allow execution).

```python
async def my_before_hook(context, cancellation):
    if context.tool_call.name == "dangerous_tool":
        return BeforeToolCallResult(block=True, reason="Tool is restricted")
    return None  # Allow execution
```

## AfterToolCallHook

The `after_tool_call` hook receives an `AfterToolCallContext` and returns an `AfterToolCallResult` (or `None` to use the original result).

```python
async def my_after_hook(context, cancellation):
    # Log all tool results
    print(f"Tool {context.tool_call.name} returned: {context.result}")
    return None  # Keep original result
```

## Configuration

Content coming soon.

## Next steps

- [Tools](tools.md) -- The AgentTool execute method
- [Control Tool Execution](../howto/control-tool-execution.md) -- Sequential vs parallel
- [Agent Lifecycle](agent-lifecycle.md) -- Configuring hooks in AgentOptions
