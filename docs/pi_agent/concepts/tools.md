# Tools

In pi-agent, tools are implemented as `AgentTool` subclasses. Each tool has a name, description, JSON Schema for parameters, and an async `execute()` method that performs the actual work.

## Defining a tool with a subclass

Subclass `AgentTool` and implement `execute()`:

```python
from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent


class WeatherTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            label="Get Weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["city"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        city = params["city"]
        # ... call an API, do computation, etc.
        return AgentToolResult(
            content=[TextContent(text=f"18C and sunny in {city}")]
        )
```

### Constructor parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier for the tool |
| `label` | `str` | Human-readable display name |
| `description` | `str` | LLM-facing description of what the tool does |
| `parameters` | `dict` | JSON Schema defining input parameters |

### execute() signature

```python
async def execute(
    self,
    tool_call_id: str,
    params: Any,
    cancellation: CancellationToken | None = None,
    on_update: AgentToolUpdateCallback | None = None,
) -> AgentToolResult
```

| Parameter | Description |
|---|---|
| `tool_call_id` | Unique ID for this invocation |
| `params` | Validated arguments matching the JSON Schema |
| `cancellation` | Token for cooperative cancellation |
| `on_update` | Callback for streaming partial results |

## Defining a tool with from_function

For simple tools, use the factory method:

```python
from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent


async def calculate(tool_call_id, params, cancellation=None, on_update=None):
    expr = params["expression"]
    result = eval(expr)  # (use a safe evaluator in production!)
    return AgentToolResult(
        content=[TextContent(text=str(result))]
    )


calc_tool = AgentTool.from_function(
    name="calculate",
    label="Calculator",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            },
        },
        "required": ["expression"],
    },
    fn=calculate,
)
```

## AgentToolResult

The result returned by `execute()`:

```python
from pi_agent import AgentToolResult
from pi_ai import TextContent, ImageContent

# Text result
result = AgentToolResult(
    content=[TextContent(text="The answer is 42")]
)

# Result with metadata
result = AgentToolResult(
    content=[TextContent(text="File saved")],
    details={"path": "/tmp/output.txt", "bytes": 1024},
)

# Image result
result = AgentToolResult(
    content=[ImageContent(data=base64_data, mime_type="image/png")]
)
```

The `content` field accepts a list of `TextContent` and/or `ImageContent` blocks. The `details` field is arbitrary metadata that is included in the tool result message.

## Streaming partial results

Long-running tools can report progress via the `on_update` callback:

```python
class ResearchTool(AgentTool):
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        query = params["query"]
        results = []

        for i, source in enumerate(sources):
            # Check for cancellation
            if cancellation and cancellation.is_cancelled:
                break

            result = await fetch_source(source, query)
            results.append(result)

            # Stream progress
            if on_update:
                on_update(AgentToolResult(
                    content=[TextContent(text=f"Searched {i+1}/{len(sources)} sources")],
                    details={"progress": (i + 1) / len(sources)},
                ))

        return AgentToolResult(
            content=[TextContent(text="\n".join(results))]
        )
```

Each `on_update` call emits a `ToolExecutionUpdateEvent` that subscribers can observe.

## Error handling

Raise any exception from `execute()` to signal failure. The agent catches it and reports the error to the LLM:

```python
class FileTool(AgentTool):
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        path = params["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        content = open(path).read()
        return AgentToolResult(content=[TextContent(text=content)])
```

The LLM sees the error message and can retry or inform the user.

## Argument pre-processing

Override `prepare_arguments()` to transform raw arguments before validation:

```python
class FlexibleTool(AgentTool):
    def prepare_arguments(self, args):
        # Normalize a string argument to the expected object shape
        if isinstance(args, str):
            return {"query": args}
        return args
```

This runs before JSON Schema validation, so the transformed arguments must match the schema.

## Converting to pi-ai Tool

`AgentTool` has a `to_tool()` method that produces a pi-ai `Tool` for inclusion in LLM context. The agent calls this internally, but you can use it directly:

```python
pi_ai_tool = my_agent_tool.to_tool()
# Tool(name="...", description="...", parameters={...})
```

## Registering tools with the agent

Pass tools in `InitialAgentState`, or modify `agent.state.tools` between runs:

```python
# At creation
agent = Agent(AgentOptions(
    initial_state=InitialAgentState(
        tools=[weather_tool, calc_tool],
        ...
    ),
))

# Between runs
agent.state.tools = [weather_tool, calc_tool, new_tool]
```

## Next steps

- [Hooks](hooks.md) -- Intercept tool calls with before/after hooks
- [Events](events.md) -- Tool execution events
- [Control Tool Execution](../howto/control-tool-execution.md) -- Sequential vs parallel
