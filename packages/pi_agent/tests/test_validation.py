"""Tests for agent tool argument validation."""

import pytest

from pi_llm_agent import AgentTool, validate_agent_tool_arguments
from pi_llm.types import ToolCall


def _tool():
    return AgentTool(
        name="get_weather",
        label="Get Weather",
        description="Get weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


def test_validate_agent_tool_arguments_passes():
    tool = _tool()
    tc = ToolCall(id="tc1", name="get_weather", arguments={"city": "LA"})
    result = validate_agent_tool_arguments(tool, tc)
    assert result == {"city": "LA"}


def test_validate_agent_tool_arguments_fails():
    tool = _tool()
    tc = ToolCall(id="tc1", name="get_weather", arguments={})
    with pytest.raises(ValueError, match="Validation failed"):
        validate_agent_tool_arguments(tool, tc)
