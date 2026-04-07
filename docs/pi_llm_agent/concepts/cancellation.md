# Cancellation

pi-llm-agent uses cooperative cancellation via `CancellationToken`. Tokens are passed to event listeners and tool `execute()` methods, allowing graceful shutdown at any point in the agent loop.

## Overview

- **`CancellationToken`** -- A token that can be checked and cancelled.
- **`agent.abort()`** -- Cancel the current agent run.
- **`cancellation.is_cancelled`** -- Check if cancellation has been requested.
- **`cancellation.cancel()`** -- Request cancellation.

## Using CancellationToken in tools

```python
class LongRunningTool(AgentTool):
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        for i in range(100):
            if cancellation and cancellation.is_cancelled:
                return AgentToolResult(
                    content=[TextContent(text="Cancelled")]
                )
            await asyncio.sleep(0.1)
        return AgentToolResult(content=[TextContent(text="Done")])
```

## Aborting the agent

```python
agent.abort()  # Cancels the current run
await agent.wait_for_idle()  # Wait for cleanup
```

## CancellationToken in event listeners

Content coming soon.

## Next steps

- [Agent Lifecycle](agent-lifecycle.md) -- The abort() and wait_for_idle() methods
- [Tools](tools.md) -- Using cancellation in tool execute()
