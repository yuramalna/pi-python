# Tools

Tools enable LLMs to interact with external systems by calling named functions with structured arguments. In pi-ai, tools are defined with a name, description, and a JSON Schema for their parameters.

## Defining a tool

The `Tool` class takes three fields:

```python
from pi_ai import Tool

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units",
            },
        },
        "required": ["city"],
    },
)
```

- **name** -- A unique identifier the LLM uses to invoke the tool.
- **description** -- A human-readable string the LLM uses to decide when to call the tool.
- **parameters** -- A JSON Schema object describing the tool's input parameters.

## Creating tools from Pydantic models

If you already have a Pydantic model for your tool's parameters, use `Tool.from_pydantic()` to generate the schema automatically:

```python
from pydantic import BaseModel, Field
from pi_ai import Tool


class WeatherParams(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(
        default="celsius",
        description="Temperature units",
        json_schema_extra={"enum": ["celsius", "fahrenheit"]},
    )


weather_tool = Tool.from_pydantic(
    name="get_weather",
    description="Get current weather for a city",
    model_class=WeatherParams,
)
```

## Adding tools to the context

Pass tools in the `Context` to make them available to the LLM:

```python
from pi_ai import Context

context = Context(
    system_prompt="Use the available tools to help the user.",
    messages=[...],
    tools=[weather_tool],
)
```

The LLM will decide whether to call a tool based on the user's request and the tool descriptions.

## Handling tool call events

When the LLM decides to call a tool, the event stream emits `ToolCallStartEvent`, zero or more `ToolCallDeltaEvent`s, and finally `ToolCallEndEvent` with the parsed arguments.

```python
from pi_ai import (
    ToolCallEndEvent,
    ToolResultMessage,
    TextContent,
    stream_simple,
    SimpleStreamOptions,
)
import time

event_stream = stream_simple(model, context, SimpleStreamOptions())

async for event in event_stream:
    if isinstance(event, ToolCallEndEvent):
        tc = event.tool_call
        print(f"Tool call: {tc.name}(id={tc.id})")
        print(f"Arguments: {tc.arguments}")

# After consuming events, get the complete message
assistant_msg = await event_stream.result()
```

## Returning tool results

After executing the tool, send back a `ToolResultMessage` to continue the conversation:

```python
import time
from pi_ai import ToolResultMessage, TextContent

result = ToolResultMessage(
    tool_call_id=tc.id,       # Must match the ToolCall.id
    tool_name=tc.name,         # Must match the ToolCall.name
    content=[
        TextContent(text='{"temp": "18C", "condition": "Sunny"}'),
    ],
    timestamp=int(time.time() * 1000),
)
```

Add both the assistant message (containing the tool call) and the tool result to the context for the next LLM call:

```python
messages.append(assistant_msg)
messages.append(result)

context = Context(
    system_prompt="Use the available tools to help the user.",
    messages=messages,
    tools=[weather_tool],
)

# Continue streaming -- the LLM will incorporate the tool result
event_stream = stream_simple(model, context, SimpleStreamOptions())
```

## ToolCall structure

The `ToolCall` object returned in `ToolCallEndEvent.tool_call` has these fields:

| Field | Type | Description |
|---|---|---|
| `type` | `Literal["toolCall"]` | Always `"toolCall"` |
| `id` | `str` | Unique ID for this invocation |
| `name` | `str` | Tool name (matches `Tool.name`) |
| `arguments` | `dict[str, Any]` | Parsed arguments matching the JSON Schema |

## Validating tool arguments

pi-ai provides utilities to validate tool call arguments against the schema:

```python
from pi_ai import validate_tool_call, validate_tool_arguments

# Validate a ToolCall against a list of Tools
errors = validate_tool_call(tool_call, [weather_tool])

# Validate raw arguments against a JSON Schema
errors = validate_tool_arguments(tool_call.arguments, weather_tool.parameters)
```

Both return a list of validation error strings (empty list means valid).

## Error results

If tool execution fails, set `is_error=True` on the result message:

```python
error_result = ToolResultMessage(
    tool_call_id=tc.id,
    tool_name=tc.name,
    content=[TextContent(text="City not found")],
    is_error=True,
    timestamp=int(time.time() * 1000),
)
```

The LLM will see the error and can retry or inform the user.

## Next steps

- [Handle Tool Calls](../howto/handle-tool-calls.md) -- Step-by-step guide
- [Events](events.md) -- The tool call event family
- [Messages & Context](messages-and-context.md) -- ToolResultMessage details
