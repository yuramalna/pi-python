"""Tool call argument validation using jsonschema.

Validates tool call arguments against their JSON Schema definitions.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import jsonschema

from pi_llm.types import Tool, ToolCall


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> Any:
    """Find a tool by name and validate the call's arguments.

    Args:
        tools: Available tools to search.
        tool_call: The tool call to validate.

    Returns:
        A deep copy of the validated arguments.

    Raises:
        ValueError: If the tool is not found or arguments are invalid.
    """
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if not tool:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments against JSON Schema.

    Returns a deep copy of validated arguments.
    Raises ValueError with formatted message on validation failure.
    """
    args = copy.deepcopy(tool_call.arguments)
    try:
        jsonschema.validate(instance=args, schema=tool.parameters)
    except jsonschema.ValidationError as e:
        raise ValueError(
            f'Validation failed for tool "{tool_call.name}":\n'
            f"  - {e.json_path}: {e.message}\n\n"
            f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2)}"
        ) from e
    return args
