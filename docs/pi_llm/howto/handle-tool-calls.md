# Handle Tool Calls

This guide walks through the complete workflow of defining a tool, streaming a response, handling tool calls, and returning results to the LLM.

## Overview

The tool call workflow has four steps:

1. Define a `Tool` with a JSON Schema for its parameters.
2. Include the tool in the `Context` and stream a response.
3. When the LLM emits a `ToolCallEndEvent`, execute the tool.
4. Return the result as a `ToolResultMessage` and continue the conversation.

## Step-by-step

Content coming soon.

## Next steps

- [Tools](../concepts/tools.md) -- Tool definition reference
- [Events](../concepts/events.md) -- The tool call event family
- [Handle Errors](handle-errors.md) -- Error handling in tool workflows
