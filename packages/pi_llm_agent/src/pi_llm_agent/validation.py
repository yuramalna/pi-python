"""Tool argument validation for the agent package."""

from __future__ import annotations

from typing import Any

from pi_llm.types import ToolCall
from pi_llm.utils.validation import validate_tool_arguments
from pi_llm_agent.types import AgentTool


def validate_agent_tool_arguments(tool: AgentTool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments against the agent tool's JSON Schema.

    Converts ``AgentTool`` to a pi_llm ``Tool`` and delegates to
    ``pi_llm.utils.validation.validate_tool_arguments``.
    """
    return validate_tool_arguments(tool.to_tool(), tool_call)
