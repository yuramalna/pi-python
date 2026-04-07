"""pi_agent — General-purpose agent framework."""

from pi_llm_agent.agent import (
    DEFAULT_MODEL,
    Agent,
    AgentOptions,
    AgentState,
    InitialAgentState,
    PendingMessageQueue,
    default_convert_to_llm,
)
from pi_llm_agent.agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)
from pi_llm_agent.cancellation import CancellationToken
from pi_llm_agent.types import (
    AfterToolCallContext,
    AfterToolCallHook,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentStartEvent,
    AgentThinkingLevel,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentToolUpdateCallback,
    BeforeToolCallContext,
    BeforeToolCallHook,
    BeforeToolCallResult,
    QueueMode,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from pi_llm_agent.validation import validate_agent_tool_arguments

__all__ = [
    # Agent class
    "Agent",
    "AgentOptions",
    "AgentState",
    "InitialAgentState",
    "PendingMessageQueue",
    "default_convert_to_llm",
    "DEFAULT_MODEL",
    # Agent loop
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
    # Cancellation
    "CancellationToken",
    # Type aliases
    "AgentThinkingLevel",
    "ToolExecutionMode",
    "QueueMode",
    "AgentToolCall",
    "AgentMessage",
    "StreamFn",
    "AgentEventSink",
    "AgentToolUpdateCallback",
    # Tool types
    "AgentToolResult",
    "AgentTool",
    # Context
    "AgentContext",
    # Hook types
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "BeforeToolCallHook",
    "AfterToolCallHook",
    # Events
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "AgentMessageStartEvent",
    "AgentMessageUpdateEvent",
    "AgentMessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "AgentEvent",
    # Config
    "AgentLoopConfig",
    # Validation
    "validate_agent_tool_arguments",
]
