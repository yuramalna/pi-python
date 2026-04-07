# pi-llm

Unified LLM provider abstraction for Python.

pi-llm provides a clean, typed interface for making LLM API calls with streaming events, tool calling, extended thinking, and cost tracking.

## Key Concepts

- [Models & Providers](concepts/models-and-providers.md) — Model catalog and provider registry
- [Streaming](concepts/streaming.md) — `stream_simple()`, `complete_simple()`, and `EventStream`
- [Events](concepts/events.md) — The 12 streaming event types
- [Messages & Context](concepts/messages-and-context.md) — Message types and conversation context
- [Tools](concepts/tools.md) — Tool definition, JSON Schema, and validation
- [Thinking / Reasoning](concepts/thinking.md) — Extended thinking levels
- [Cost Tracking](concepts/cost-tracking.md) — Token usage and dollar cost calculation

## How-to Guides

- [Handle Tool Calls](howto/handle-tool-calls.md)
- [Add Image Input](howto/add-image-input.md)
- [Handle Errors](howto/handle-errors.md)

## API Reference

See the auto-generated [API Reference](reference/) for complete documentation of all public types and functions.
