"""Tool argument validation for the agent package."""

from __future__ import annotations

from typing import Any

from pi_agent.types import AgentTool
from pi_ai.types import ToolCall
from pi_ai.utils.validation import validate_tool_arguments


def validate_agent_tool_arguments(tool: AgentTool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments against the agent tool's JSON Schema.

    Converts ``AgentTool`` to a pi_ai ``Tool`` and delegates to
    ``pi_ai.utils.validation.validate_tool_arguments``.
    """
    return validate_tool_arguments(tool.to_tool(), tool_call)
