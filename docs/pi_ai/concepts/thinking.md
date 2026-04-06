# Thinking / Reasoning

Some LLM models support extended thinking (also called reasoning), where the model explicitly works through a problem before producing its final answer. pi-ai provides first-class support for controlling reasoning levels and observing thinking events.

## ThinkingLevel

The `ThinkingLevel` type defines five levels of reasoning effort:

| Level | Description |
|---|---|
| `"minimal"` | Minimal reasoning |
| `"low"` | Low reasoning effort |
| `"medium"` | Moderate reasoning effort |
| `"high"` | High reasoning effort |
| `"xhigh"` | Maximum reasoning effort (not supported by all models) |

Pass the level via `SimpleStreamOptions`:

```python
from pi_ai import SimpleStreamOptions

options = SimpleStreamOptions(reasoning="medium")
```

## Thinking events

When reasoning is enabled, the event stream includes `ThinkingStartEvent`, `ThinkingDeltaEvent`, and `ThinkingEndEvent` events alongside the regular text events.

## ThinkingContent

Reasoning output is stored as `ThinkingContent` blocks in the `AssistantMessage.content` list, alongside `TextContent` and `ToolCall` blocks.

## Thinking budgets

Content coming soon.

## Next steps

- [Streaming](streaming.md) -- Passing reasoning options
- [Events](events.md) -- The thinking event family
- [Cost Tracking](cost-tracking.md) -- Reasoning tokens and cost
