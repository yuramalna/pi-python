# Steering & Follow-up

Steering and follow-up queues let you inject messages into a running agent's conversation. Steering messages are delivered between turns; follow-up messages are delivered after the current run completes.

## Overview

- **Steering** -- Use `agent.steer(message)` to inject a message between turns. The agent sees it before its next LLM call.
- **Follow-up** -- Use `agent.follow_up(message)` to queue a message for after the current run. The agent processes it as a new prompt.

## Queue modes

Both queues support two drain modes:

| Mode | Behavior |
|---|---|
| `"one-at-a-time"` | Process one queued message per turn (default) |
| `"all"` | Process all queued messages at once |

```python
agent.steering_mode = "all"
agent.follow_up_mode = "one-at-a-time"
```

## steer()

```python
agent.steer(UserMessage(
    content="Focus on cost analysis.",
    timestamp=int(time.time() * 1000),
))
```

## follow_up()

```python
agent.follow_up(UserMessage(
    content="Now summarize your findings.",
    timestamp=int(time.time() * 1000),
))
```

## Queue management

Content coming soon.

## Next steps

- [Agent Lifecycle](agent-lifecycle.md) -- Queue configuration in AgentOptions
- [Cancellation](cancellation.md) -- Aborting a run
- [Agent Loop](agent-loop.md) -- How messages are polled between turns
