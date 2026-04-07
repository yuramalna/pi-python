# Cost Tracking

pi-llm tracks token usage and calculates dollar costs for every LLM request. The `Usage` and `CostBreakdown` types provide detailed breakdowns of input, output, and cache tokens.

## Key types

- **`Usage`** -- Token counts for a single request (input, output, cache read, cache write, total).
- **`CostBreakdown`** -- Dollar costs for each token category (input, output, cache read, cache write, total).
- **`ModelCost`** -- Pricing per million tokens for a model.
- **`calculate_cost(usage, model)`** -- Compute the dollar cost from usage and model pricing.

## Usage

Every `AssistantMessage` includes a `usage` field with token counts:

```python
message = await complete_simple(model, context, options)
print(f"Input tokens: {message.usage.input}")
print(f"Output tokens: {message.usage.output}")
print(f"Total tokens: {message.usage.total_tokens}")
```

## CostBreakdown

The `usage.cost` field contains the dollar breakdown:

```python
cost = message.usage.cost
print(f"Input cost:  ${cost.input:.6f}")
print(f"Output cost: ${cost.output:.6f}")
print(f"Total cost:  ${cost.total:.6f}")
```

## calculate_cost

Content coming soon.

## Next steps

- [Models & Providers](models-and-providers.md) -- Model pricing data
- [Streaming](streaming.md) -- Getting usage from streamed responses
