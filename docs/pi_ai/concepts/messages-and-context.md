# Messages & Context

pi-llm uses typed message objects to represent a conversation. The `Context` object bundles messages with a system prompt and tool definitions, forming the complete input for an LLM call.

## Message types

There are three message types, discriminated by their `role` field:

- **`UserMessage`** (`role="user"`) -- A message from the user. Content can be a plain string or a list of `TextContent` / `ImageContent` blocks.
- **`AssistantMessage`** (`role="assistant"`) -- A response from the LLM. Content is a list of `TextContent`, `ThinkingContent`, and/or `ToolCall` blocks.
- **`ToolResultMessage`** (`role="toolResult"`) -- The result of executing a tool, sent back to the LLM.

The `Message` union type covers all three: `UserMessage | AssistantMessage | ToolResultMessage`.

## Context

The `Context` class bundles everything needed for an LLM call:

| Field | Type | Description |
|---|---|---|
| `system_prompt` | `str` | System-level instructions |
| `messages` | `list[Message]` | Conversation history |
| `tools` | `list[Tool]` | Available tools |

## Content blocks

Content coming soon.

## Serialization

Content coming soon.

## Next steps

- [Events](events.md) -- How messages are built incrementally via events
- [Tools](tools.md) -- The Tool definitions in Context
