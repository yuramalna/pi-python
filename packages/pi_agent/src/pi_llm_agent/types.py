"""Agent type definitions.

Tools, events, hooks, loop configuration, and type aliases for the agent
framework.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

from pi_llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Model,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)

if TYPE_CHECKING:
    from pi_llm.utils.event_stream import AssistantMessageEventStream
    from pi_llm_agent.cancellation import CancellationToken

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
"""Extended thinking level for the agent. Includes ``"off"`` (pi_llm's
``ThinkingLevel`` does not)."""

ToolExecutionMode = Literal["sequential", "parallel"]
"""How tool calls within a turn are executed."""

QueueMode = Literal["all", "one-at-a-time"]
"""How queued messages are drained: all at once or one per turn."""

AgentToolCall = ToolCall
"""A single tool call content block from an assistant message."""

AgentMessage = Any
"""Union of LLM messages and custom application messages."""

# ---------------------------------------------------------------------------
# AgentToolResult
# ---------------------------------------------------------------------------


@dataclass
class AgentToolResult:
    """Result returned by a tool's ``execute()`` method.

    Also used for partial progress updates via ``on_update``.

    Attributes:
        content: Result content blocks (text or images).
        details: Arbitrary metadata about the execution.
    """

    content: list[TextContent | ImageContent]
    details: Any = None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

AgentToolUpdateCallback = Callable[["AgentToolResult"], None]
"""Callback passed to ``AgentTool.execute()`` for streaming partial updates."""

# ---------------------------------------------------------------------------
# AgentTool
# ---------------------------------------------------------------------------


class AgentTool:
    """Base class for tools that the agent can invoke.

    Subclass and implement ``execute()``, or use the ``from_function()``
    factory for simple cases.

    Attributes:
        name: Unique identifier for the tool.
        label: Human-readable display name.
        description: LLM-facing description of what the tool does.
        parameters: JSON Schema defining the tool's input parameters.

    Example:
        >>> class WeatherTool(AgentTool):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="get_weather",
        ...             label="Get Weather",
        ...             description="Get current weather for a city",
        ...             parameters={
        ...                 "type": "object",
        ...                 "properties": {"city": {"type": "string"}},
        ...                 "required": ["city"],
        ...             },
        ...         )
        ...
        ...     async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        ...         return AgentToolResult(content=[TextContent(text=f"Sunny in {params['city']}")])
    """

    def __init__(
        self,
        name: str,
        label: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.label = label
        self.description = description
        self.parameters = parameters

    def prepare_arguments(self, args: Any) -> Any:
        """Optional pre-processing of raw arguments before validation.

        Override to transform arguments before they are validated against
        the JSON Schema. Default implementation is a passthrough.
        """
        return args

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancellation: CancellationToken | None = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """Execute the tool with validated parameters.

        Args:
            tool_call_id: Unique ID for this tool call.
            params: Validated arguments matching the tool's JSON Schema.
            cancellation: Token to check for cooperative cancellation.
            on_update: Callback for streaming partial results.

        Returns:
            The tool execution result.

        Raises:
            Exception: Raise any exception to signal tool failure. The agent
                will report it to the LLM as an error.
        """
        raise NotImplementedError

    def to_tool(self) -> Tool:
        """Convert to a pi_llm ``Tool`` for inclusion in LLM context."""
        return Tool(name=self.name, description=self.description, parameters=self.parameters)

    @classmethod
    def from_function(
        cls,
        name: str,
        label: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Awaitable[AgentToolResult]],
    ) -> AgentTool:
        """Create a tool from a JSON Schema and an async function.

        Args:
            name: Tool name.
            label: Display name.
            description: LLM-facing description.
            parameters: JSON Schema for parameters.
            fn: Async function with the same signature as ``execute()``.

        Returns:
            A new ``AgentTool`` instance.
        """
        instance = cls(name=name, label=label, description=description, parameters=parameters)
        instance.execute = fn  # type: ignore[assignment]
        return instance


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Context snapshot passed into the low-level agent loop.

    Attributes:
        system_prompt: System-level instructions.
        messages: Conversation history (agent messages).
        tools: Available tools, or ``None`` for no tools.
    """

    system_prompt: str = ""
    messages: list[Any] = field(default_factory=list)
    tools: list[AgentTool] | None = None


# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------


@dataclass
class BeforeToolCallContext:
    """Context passed to the ``before_tool_call`` hook.

    Attributes:
        assistant_message: The assistant message that requested the tool call.
        tool_call: The tool call being preflighted.
        args: Validated arguments (after ``prepare_arguments`` and schema validation).
        context: Current agent context snapshot.
    """

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


@dataclass
class BeforeToolCallResult:
    """Result from the ``before_tool_call`` hook.

    Attributes:
        block: Set to ``True`` to prevent tool execution.
        reason: Human-readable reason for blocking (reported to the LLM).
    """

    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallContext:
    """Context passed to the ``after_tool_call`` hook.

    Attributes:
        assistant_message: The assistant message that requested the tool call.
        tool_call: The tool call that was executed.
        args: Validated arguments that were passed to the tool.
        result: The tool execution result.
        is_error: Whether the tool execution failed.
        context: Current agent context snapshot.
    """

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class AfterToolCallResult:
    """Partial override returned by the ``after_tool_call`` hook.

    Only non-``None`` fields override the original result.

    Attributes:
        content: Override result content.
        details: Override result details.
        is_error: Override error flag.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


BeforeToolCallHook = Callable[
    ["BeforeToolCallContext", "CancellationToken | None"],
    Awaitable[BeforeToolCallResult | None],
]
"""Async hook called before each tool execution. Return a result with
``block=True`` to prevent execution."""

AfterToolCallHook = Callable[
    ["AfterToolCallContext", "CancellationToken | None"],
    Awaitable[AfterToolCallResult | None],
]
"""Async hook called after each tool execution. Return a result to
override content, details, or error flag."""


# ---------------------------------------------------------------------------
# StreamFn / AgentEventSink
# ---------------------------------------------------------------------------

StreamFn = Callable[
    [Model, Any, Any],
    "AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]",
]
"""Stream function used by the agent loop. Matches the ``stream_simple`` signature."""

# AgentEventSink is defined after AgentEvent (below).

# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentStartEvent:
    """Emitted when the agent loop begins processing."""

    type: str = field(default="agent_start", init=False)


@dataclass(slots=True)
class AgentEndEvent:
    """Emitted when the agent loop finishes. Contains all new messages.

    Attributes:
        messages: All messages added during this agent run.
    """

    messages: list[Any]
    type: str = field(default="agent_end", init=False)


@dataclass(slots=True)
class TurnStartEvent:
    """Emitted at the start of each conversation turn."""

    type: str = field(default="turn_start", init=False)


@dataclass(slots=True)
class TurnEndEvent:
    """Emitted at the end of a turn (one LLM call + tool executions).

    Attributes:
        message: The assistant message from this turn.
        tool_results: Tool result messages produced in this turn.
    """

    message: Any
    tool_results: list[ToolResultMessage]
    type: str = field(default="turn_end", init=False)


@dataclass(slots=True)
class AgentMessageStartEvent:
    """Emitted when any message (user, assistant, toolResult) begins."""

    message: Any
    type: str = field(default="message_start", init=False)


@dataclass(slots=True)
class AgentMessageUpdateEvent:
    """Emitted for incremental assistant message streaming updates.

    Attributes:
        message: The partial assistant message being built.
        assistant_message_event: The underlying pi_llm streaming event.
    """

    message: Any
    assistant_message_event: AssistantMessageEvent
    type: str = field(default="message_update", init=False)


@dataclass(slots=True)
class AgentMessageEndEvent:
    """Emitted when a message is complete and added to the transcript."""

    message: Any
    type: str = field(default="message_end", init=False)


@dataclass(slots=True)
class ToolExecutionStartEvent:
    """Emitted when a tool call begins execution.

    Attributes:
        tool_call_id: Unique ID of the tool call.
        tool_name: Name of the tool being executed.
        args: Raw arguments from the LLM.
    """

    tool_call_id: str
    tool_name: str
    args: Any
    type: str = field(default="tool_execution_start", init=False)


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    """Emitted when a tool streams a partial result via ``on_update``.

    Attributes:
        tool_call_id: Unique ID of the tool call.
        tool_name: Name of the tool.
        args: Arguments passed to the tool.
        partial_result: The partial result from ``on_update``.
    """

    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: str = field(default="tool_execution_update", init=False)


@dataclass(slots=True)
class ToolExecutionEndEvent:
    """Emitted when a tool call completes execution.

    Attributes:
        tool_call_id: Unique ID of the tool call.
        tool_name: Name of the tool.
        result: The final tool result.
        is_error: Whether the execution failed.
    """

    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: str = field(default="tool_execution_end", init=False)


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentMessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]
"""Union of all agent event types."""

AgentEventSink = Callable[[AgentEvent], "Awaitable[None] | None"]
"""Callback that receives agent events. Can be sync or async."""


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentLoopConfig:
    """Configuration for the low-level agent loop.

    Attributes:
        model: LLM model to use.
        convert_to_llm: Function to filter agent messages to LLM messages.
        transform_context: Optional pre-processing of messages before LLM call.
        get_api_key: Resolve API key by provider name.
        get_steering_messages: Poll for steering messages between turns.
        get_follow_up_messages: Poll for follow-up messages after all turns.
        before_tool_call: Hook called before each tool execution.
        after_tool_call: Hook called after each tool execution.
        tool_execution: ``"parallel"`` (default) or ``"sequential"``.
        reasoning: Extended thinking level.
        api_key: Static API key (``get_api_key`` takes precedence).
        session_id: Session ID for prompt caching.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        max_retry_delay_ms: Cap on retry backoff.
        thinking_budgets: Custom token budgets per thinking level.
        on_payload: Hook called with raw API params.
        cache_retention: Prompt cache retention preference.
        headers: Additional HTTP headers.
        metadata: Arbitrary metadata passed to the provider.
    """

    # -- Required --
    model: Model
    convert_to_llm: Callable[[list[Any]], Awaitable[list[Any]] | list[Any]]

    # -- Optional callbacks --
    transform_context: (
        Callable[[list[Any], CancellationToken | None], Awaitable[list[Any]]] | None
    ) = None
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
    get_steering_messages: Callable[[], Awaitable[list[Any]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[Any]]] | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None

    # -- Execution mode --
    tool_execution: ToolExecutionMode = "parallel"

    # -- Inlined SimpleStreamOptions fields --
    reasoning: AgentThinkingLevel | None = None
    api_key: str | None = None
    session_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retry_delay_ms: int | None = None
    thinking_budgets: dict[str, int] | None = None
    on_payload: Callable[..., Any] | None = None
    cache_retention: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
