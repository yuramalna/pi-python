"""Stateful Agent wrapper — the primary public API.

The ``Agent`` class manages conversation state, tool execution, event
dispatch, and message queueing. It wraps the low-level agent loop.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_llm.stream import stream_simple
from pi_llm.types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    Usage,
    UserMessage,
)
from pi_llm_agent.agent_loop import run_agent_loop, run_agent_loop_continue
from pi_llm_agent.cancellation import CancellationToken
from pi_llm_agent.types import (
    AfterToolCallHook,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessageEndEvent,
    AgentMessageStartEvent,
    AgentMessageUpdateEvent,
    AgentThinkingLevel,
    AgentTool,
    BeforeToolCallHook,
    QueueMode,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    TurnEndEvent,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def default_convert_to_llm(messages: list[Any]) -> list[Any]:
    """Default message filter: passes through user, assistant, and toolResult messages."""
    return [
        m
        for m in messages
        if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")
    ]


DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    input_types=[],
    cost=None,
    context_window=0,
    max_tokens=0,
)


# ---------------------------------------------------------------------------
# PendingMessageQueue
# ---------------------------------------------------------------------------


class PendingMessageQueue:
    """Message queue with configurable drain mode.

    In ``"all"`` mode, ``drain()`` returns all queued messages at once.
    In ``"one-at-a-time"`` mode, ``drain()`` returns only the first message.
    """

    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[Any] = []

    def enqueue(self, message: Any) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[Any]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


# ---------------------------------------------------------------------------
# InitialAgentState / AgentOptions
# ---------------------------------------------------------------------------


@dataclass
class InitialAgentState:
    """Initial state for constructing an Agent.

    Attributes:
        system_prompt: System-level instructions for the LLM.
        model: LLM model to use.
        thinking_level: Extended thinking level (default ``"off"``).
        tools: Available tools.
        messages: Pre-existing conversation history.
    """

    system_prompt: str = ""
    model: Model | None = None
    thinking_level: AgentThinkingLevel = "off"
    tools: list[AgentTool] | None = None
    messages: list[Any] | None = None


@dataclass
class AgentOptions:
    """Options for constructing an ``Agent``.

    Attributes:
        initial_state: Initial agent state (model, tools, system prompt).
        convert_to_llm: Custom message filter for LLM calls.
        transform_context: Pre-process messages before each LLM call.
        stream_fn: Custom streaming function (default: ``stream_simple``).
        get_api_key: Dynamic API key resolver.
        on_payload: Hook called with raw API params before each request.
        before_tool_call: Hook called before each tool execution.
        after_tool_call: Hook called after each tool execution.
        steering_mode: How steering queue drains (default ``"one-at-a-time"``).
        follow_up_mode: How follow-up queue drains (default ``"one-at-a-time"``).
        session_id: Session ID for prompt caching.
        thinking_budgets: Custom token budgets per thinking level.
        max_retry_delay_ms: Cap on retry backoff.
        tool_execution: ``"parallel"`` (default) or ``"sequential"``.
    """

    initial_state: InitialAgentState | None = None
    convert_to_llm: Callable[[list[Any]], list[Any] | Awaitable[list[Any]]] | None = (
        None
    )
    transform_context: (
        Callable[[list[Any], CancellationToken | None], Awaitable[list[Any]]] | None
    ) = None
    stream_fn: StreamFn | None = None
    get_api_key: (
        Callable[[str], str | None | Awaitable[str | None]] | None
    ) = None
    on_payload: Callable[..., Any] | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    session_id: str | None = None
    thinking_budgets: dict[str, int] | None = None
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode = "parallel"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class AgentState:
    """Mutable agent state with copy-on-assign for tools and messages.

    Access via ``agent.state``. Assigning to ``tools`` or ``messages``
    creates a shallow copy, preventing external mutation.

    Attributes:
        system_prompt: System-level instructions.
        model: Current LLM model.
        thinking_level: Current extended thinking level.
        tools: Available tools (copy-on-assign).
        messages: Conversation transcript (copy-on-assign).
        is_streaming: Whether the agent is currently processing.
        streaming_message: Partial message being streamed, or ``None``.
        pending_tool_calls: Set of tool call IDs awaiting completion.
        error_message: Last error message, or ``None``.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: AgentThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[Any] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model: Model = model or DEFAULT_MODEL
        self.thinking_level: AgentThinkingLevel = thinking_level
        self._tools: list[AgentTool] = list(tools) if tools else []
        self._messages: list[Any] = list(messages) if messages else []
        # Runtime state (mutated only by Agent internals)
        self.is_streaming: bool = False
        self.streaming_message: Any | None = None
        self.pending_tool_calls: set[str] = set()
        self.error_message: str | None = None

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[Any]:
        return self._messages

    @messages.setter
    def messages(self, value: list[Any]) -> None:
        self._messages = list(value)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class Agent:
    """Stateful agent with tool execution and event streaming.

    The Agent wraps the low-level agent loop, managing the conversation
    transcript, tool execution lifecycle, event dispatch, and message
    queueing for steering and follow-up.

    Example:
        >>> agent = Agent(AgentOptions(
        ...     initial_state=InitialAgentState(
        ...         model=get_model("openai", "gpt-4o"),
        ...         system_prompt="You are helpful.",
        ...         tools=[my_tool],
        ...     ),
        ...     stream_fn=stream_simple,
        ...     get_api_key=lambda _: os.environ["OPENAI_API_KEY"],
        ... ))
        >>> agent.subscribe(lambda event, cancel: print(event.type))
        >>> await agent.prompt("Hello!")
    """

    def __init__(self, options: AgentOptions | None = None) -> None:
        opts = options or AgentOptions()
        init = opts.initial_state

        self._state = AgentState(
            system_prompt=init.system_prompt if init else "",
            model=init.model if init else None,
            thinking_level=init.thinking_level if init else "off",
            tools=init.tools if init else None,
            messages=init.messages if init else None,
        )

        # Callbacks
        self.convert_to_llm: Callable = opts.convert_to_llm or default_convert_to_llm
        self.transform_context = opts.transform_context
        self.stream_fn: StreamFn = opts.stream_fn or stream_simple
        self.get_api_key = opts.get_api_key
        self.on_payload = opts.on_payload
        self.before_tool_call = opts.before_tool_call
        self.after_tool_call = opts.after_tool_call

        # Queues
        self._steering_queue = PendingMessageQueue(opts.steering_mode)
        self._follow_up_queue = PendingMessageQueue(opts.follow_up_mode)

        # Config passthrough
        self.session_id = opts.session_id
        self.thinking_budgets = opts.thinking_budgets
        self.max_retry_delay_ms = opts.max_retry_delay_ms
        self.tool_execution: ToolExecutionMode = opts.tool_execution

        # Runtime
        self._listeners: list[Callable] = []
        self._cancellation: CancellationToken | None = None
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # starts idle

    # -- State property --

    @property
    def state(self) -> AgentState:
        return self._state

    # -- Queue mode properties --

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    # -- Queue methods --

    def steer(self, message: Any) -> None:
        """Queue a steering message to interrupt between turns."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: Any) -> None:
        """Queue a follow-up message for after the current run."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """Clear all pending steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Clear all pending follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Clear both steering and follow-up queues."""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """Check if either queue has pending messages."""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # -- Cancellation / lifecycle --

    @property
    def cancellation(self) -> CancellationToken | None:
        """The current cancellation token, or ``None`` if idle."""
        return self._cancellation

    def abort(self) -> None:
        """Cancel the current processing run."""
        if self._cancellation:
            self._cancellation.cancel()

    async def wait_for_idle(self) -> None:
        """Wait until the agent finishes processing."""
        await self._idle_event.wait()

    def reset(self) -> None:
        """Clear all messages, state, and queues."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    # -- Subscribe --

    def subscribe(
        self,
        listener: Callable[[AgentEvent, CancellationToken], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """Subscribe to agent events.

        Args:
            listener: Callback receiving ``(event, cancellation_token)``.
                Can be sync or async.

        Returns:
            An unsubscribe function. Call it to stop receiving events.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    # -- Prompt / Continue --

    async def prompt(
        self,
        input: str | Any | list[Any],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Send a prompt to the agent.

        Args:
            input: A text string, a message object, or a list of messages.
            images: Optional images to include with a text prompt.

        Raises:
            RuntimeError: If the agent is already processing.
        """
        if self._cancellation is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() "
                "to queue messages, or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """Continue from the current conversation state.

        Resumes processing without adding a new message. The last message
        must be ``user`` or ``toolResult`` (not ``assistant``).

        Raises:
            RuntimeError: If the agent is already processing or has no messages.
        """
        if self._cancellation is not None:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )

        messages = self._state.messages
        if not messages:
            raise RuntimeError("No messages to continue from")

        last_message = messages[-1]

        if hasattr(last_message, "role") and last_message.role == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(
                    queued_steering, skip_initial_steering_poll=True
                )
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # -- Private methods --

    def _normalize_prompt_input(
        self,
        input: str | Any | list[Any],
        images: list[ImageContent] | None = None,
    ) -> list[Any]:
        if isinstance(input, list):
            return input
        if not isinstance(input, str):
            return [input]
        content: list[TextContent | ImageContent] = [TextContent(text=input)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=int(time.time() * 1000))]

    async def _run_prompt_messages(
        self,
        messages: list[Any],
        skip_initial_steering_poll: bool = False,
    ) -> None:
        async def executor(cancellation: CancellationToken) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(
                    skip_initial_steering_poll=skip_initial_steering_poll
                ),
                self._process_events,
                cancellation,
                self.stream_fn,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(cancellation: CancellationToken) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                cancellation,
                self.stream_fn,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(
        self, skip_initial_steering_poll: bool = False
    ) -> AgentLoopConfig:
        _skip = skip_initial_steering_poll

        async def get_steering_messages() -> list[Any]:
            nonlocal _skip
            if _skip:
                _skip = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> list[Any]:
            return self._follow_up_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=(
                self._state.thinking_level
                if self._state.thinking_level != "off"
                else None
            ),
            session_id=self.session_id,
            on_payload=self.on_payload,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(
        self, executor: Callable[[CancellationToken], Awaitable[None]]
    ) -> None:
        if self._cancellation is not None:
            raise RuntimeError("Agent is already processing.")

        cancellation = CancellationToken()
        self._cancellation = cancellation
        self._idle_event.clear()

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(cancellation)
        except Exception as error:
            await self._handle_run_failure(error, cancellation.is_cancelled)
        finally:
            self._finish_run()

    async def _handle_run_failure(
        self, error: Exception, aborted: bool
    ) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=Usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        self._state.messages.append(failure_message)
        self._state.error_message = failure_message.error_message
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._cancellation = None
        self._idle_event.set()

    async def _process_events(self, event: AgentEvent) -> None:
        # State reduction
        if isinstance(event, AgentMessageStartEvent) or isinstance(event, AgentMessageUpdateEvent):
            self._state.streaming_message = event.message
        elif isinstance(event, AgentMessageEndEvent):
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            pending = set(self._state.pending_tool_calls)
            pending.add(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif isinstance(event, ToolExecutionEndEvent):
            pending = set(self._state.pending_tool_calls)
            pending.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending
        elif isinstance(event, TurnEndEvent):
            msg = event.message
            if (
                hasattr(msg, "role")
                and msg.role == "assistant"
                and hasattr(msg, "error_message")
                and msg.error_message
            ):
                self._state.error_message = msg.error_message
        elif isinstance(event, AgentEndEvent):
            self._state.streaming_message = None

        # Notify listeners
        cancellation = self._cancellation
        if cancellation is None:
            raise RuntimeError("Agent listener invoked outside active run")
        for listener in self._listeners:
            result = listener(event, cancellation)
            if inspect.isawaitable(result):
                await result
