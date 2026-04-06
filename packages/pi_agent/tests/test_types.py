"""Tests for agent type definitions."""

import pytest

from pi_agent import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentLoopConfig,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi_ai.types import (
    AssistantMessage,
    TextContent,
    Tool,
    ToolCall,
)

# -- AgentToolResult --


def test_agent_tool_result_construction():
    result = AgentToolResult(content=[TextContent(text="hello")])
    assert result.content[0].text == "hello"
    assert result.details is None


def test_agent_tool_result_with_details():
    result = AgentToolResult(
        content=[TextContent(text="ok")],
        details={"key": "value"},
    )
    assert result.details == {"key": "value"}


# -- AgentTool --


class _EchoTool(AgentTool):
    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo: {params}")])


async def test_agent_tool_subclass_execute():
    tool = _EchoTool(
        name="echo", label="Echo", description="Echoes input",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    result = await tool.execute("tc1", {"msg": "hi"})
    assert result.content[0].text == "echo: {'msg': 'hi'}"


async def test_agent_tool_from_function_factory():
    async def search_fn(tool_call_id, params, **_):
        return AgentToolResult(content=[TextContent(text=f"found: {params['query']}")])

    tool = AgentTool.from_function(
        "search", "Search", "Search the web",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        search_fn,
    )
    assert tool.name == "search"
    assert tool.label == "Search"
    result = await tool.execute("tc1", {"query": "python"})
    assert result.content[0].text == "found: python"


def test_agent_tool_to_tool():
    tool = _EchoTool(
        name="echo", label="Echo", description="Echoes input",
        parameters={"type": "object"},
    )
    pi_tool = tool.to_tool()
    assert isinstance(pi_tool, Tool)
    assert pi_tool.name == "echo"
    assert pi_tool.description == "Echoes input"
    assert pi_tool.parameters == {"type": "object"}


def test_agent_tool_prepare_arguments_default():
    tool = _EchoTool(
        name="echo", label="Echo", description="d", parameters={},
    )
    args = {"msg": "hi"}
    assert tool.prepare_arguments(args) is args  # passthrough


async def test_agent_tool_base_raises():
    tool = AgentTool(name="base", label="Base", description="d", parameters={})
    with pytest.raises(NotImplementedError):
        await tool.execute("tc1", {})


# -- AgentContext --


def test_agent_context_defaults():
    ctx = AgentContext()
    assert ctx.system_prompt == ""
    assert ctx.messages == []
    assert ctx.tools is None


# -- Hook types --


def test_before_tool_call_result_defaults():
    result = BeforeToolCallResult()
    assert result.block is False
    assert result.reason is None


def test_after_tool_call_result_defaults():
    result = AfterToolCallResult()
    assert result.content is None
    assert result.details is None
    assert result.is_error is None


def test_before_tool_call_context_construction():
    ctx = BeforeToolCallContext(
        assistant_message=AssistantMessage(timestamp=0),
        tool_call=ToolCall(id="tc1", name="fn", arguments={}),
        args={},
        context=AgentContext(),
    )
    assert ctx.tool_call.name == "fn"


def test_after_tool_call_context_construction():
    ctx = AfterToolCallContext(
        assistant_message=AssistantMessage(timestamp=0),
        tool_call=ToolCall(id="tc1", name="fn", arguments={}),
        args={},
        result=AgentToolResult(content=[]),
        is_error=False,
        context=AgentContext(),
    )
    assert ctx.is_error is False


# -- AgentLoopConfig --


def test_agent_loop_config_defaults():
    from pi_ai.types import Model

    model = Model(id="gpt-4o", name="gpt-4o", api="openai-responses", provider="openai")
    config = AgentLoopConfig(model=model, convert_to_llm=lambda m: m)
    assert config.tool_execution == "parallel"
    assert config.reasoning is None
    assert config.api_key is None
    assert config.temperature is None


# -- Events --


def test_agent_start_event():
    e = AgentStartEvent()
    assert e.type == "agent_start"


def test_agent_end_event():
    e = AgentEndEvent(messages=[])
    assert e.type == "agent_end"
    assert e.messages == []


def test_turn_start_event():
    e = TurnStartEvent()
    assert e.type == "turn_start"


def test_turn_end_event():
    msg = AssistantMessage(timestamp=0)
    e = TurnEndEvent(message=msg, tool_results=[])
    assert e.type == "turn_end"
    assert e.tool_results == []


def test_message_start_event():
    e = AgentMessageStartEvent(message="hello")
    assert e.type == "message_start"


def test_message_update_event():
    from pi_ai.types import StartEvent
    e = AgentMessageUpdateEvent(
        message="hello",
        assistant_message_event=StartEvent(partial=AssistantMessage(timestamp=0)),
    )
    assert e.type == "message_update"


def test_message_end_event():
    e = AgentMessageEndEvent(message="hello")
    assert e.type == "message_end"


def test_tool_execution_start_event():
    e = ToolExecutionStartEvent(tool_call_id="tc1", tool_name="fn", args={})
    assert e.type == "tool_execution_start"


def test_tool_execution_update_event():
    e = ToolExecutionUpdateEvent(
        tool_call_id="tc1", tool_name="fn", args={}, partial_result="partial",
    )
    assert e.type == "tool_execution_update"


def test_tool_execution_end_event():
    e = ToolExecutionEndEvent(
        tool_call_id="tc1", tool_name="fn", result="done", is_error=False,
    )
    assert e.type == "tool_execution_end"
    assert e.is_error is False
