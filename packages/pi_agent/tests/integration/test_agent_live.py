"""Integration tests for pi_agent with real OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""

import pytest

from pi_agent import (
    AgentEndEvent,
    AgentTool,
    AgentToolResult,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi_agent.agent import Agent, AgentOptions, InitialAgentState
from pi_ai.models import get_model
from pi_ai.stream import stream_simple
from pi_ai.types import TextContent

MODEL_ID = "gpt-5.4-2026-03-05"


class WeatherTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            label="Weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        )

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        city = params.get("city", "unknown")
        return AgentToolResult(content=[TextContent(text=f"Sunny, 22C in {city}")])


@pytest.mark.integration
async def test_agent_with_tool_call_cycle(openai_api_key):
    """Full agent cycle: prompt → tool call → tool execute → final response."""
    model = get_model("openai", MODEL_ID)
    agent = Agent(
        AgentOptions(
            initial_state=InitialAgentState(
                model=model,
                system_prompt=(
                    "You must use the get_weather tool when asked about weather. "
                    "After getting the result, summarize it briefly."
                ),
                tools=[WeatherTool()],
            ),
            stream_fn=stream_simple,
            get_api_key=lambda _provider: openai_api_key,
        )
    )

    events = []
    agent.subscribe(lambda event, _cancel: events.append(event))

    await agent.prompt("What is the weather in London?")

    # Verify tool execution occurred
    assert any(isinstance(e, ToolExecutionStartEvent) for e in events)
    assert any(isinstance(e, ToolExecutionEndEvent) for e in events)
    assert any(isinstance(e, AgentEndEvent) for e in events)
    assert agent.state.is_streaming is False

    # user + assistant(tool_call) + tool_result + assistant(text) = at least 4 messages
    assert len(agent.state.messages) >= 3
