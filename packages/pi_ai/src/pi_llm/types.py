"""Core type definitions for pi_ai.

Pydantic models, type aliases, streaming events, and configuration dataclasses
used throughout the pi_ai library.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class BaseModelWithAliases(BaseModel):
    """Base for all Pydantic models. Accepts both snake_case and camelCase."""

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
"""Why an LLM response ended."""

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
"""Extended thinking / reasoning effort level.

Note: ``"off"`` is only available in the agent layer (see ``AgentThinkingLevel``).
"""

CacheRetention = Literal["none", "short", "long"]
"""Prompt cache retention preference."""

# ---------------------------------------------------------------------------
# Cost and Usage
# ---------------------------------------------------------------------------


class CostBreakdown(BaseModelWithAliases):
    """Dollar cost breakdown for a single LLM request.

    Attributes:
        input: Cost for input tokens.
        output: Cost for output tokens.
        cache_read: Cost for tokens read from prompt cache.
        cache_write: Cost for tokens written to prompt cache.
        total: Sum of all cost components.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(BaseModelWithAliases):
    """Token usage for a single LLM request.

    Attributes:
        input: Number of input tokens.
        output: Number of output tokens.
        cache_read: Tokens read from prompt cache.
        cache_write: Tokens written to prompt cache.
        total_tokens: Sum of all token counts.
        cost: Dollar cost breakdown (populated by ``calculate_cost``).
    """

    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: CostBreakdown = Field(default_factory=CostBreakdown)


class ModelCost(BaseModelWithAliases):
    """Cost per million tokens for a model.

    Attributes:
        input: Cost per 1M input tokens.
        output: Cost per 1M output tokens.
        cache_read: Cost per 1M cached-read tokens.
        cache_write: Cost per 1M cached-write tokens.
    """

    input: float
    output: float
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextContent(BaseModelWithAliases):
    """A block of text content within a message.

    Attributes:
        type: Always ``"text"``.
        text: The text content.
        text_signature: Opaque signature for cross-provider handoff.
    """

    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(default=None, alias="textSignature")


class ThinkingContent(BaseModelWithAliases):
    """A block of model reasoning / "thinking" content.

    Attributes:
        type: Always ``"thinking"``.
        thinking: The reasoning text.
        thinking_signature: Opaque signature for cross-provider handoff.
        redacted: Whether the content was redacted by the provider.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")
    redacted: bool | None = None


class ImageContent(BaseModelWithAliases):
    """A base64-encoded image within a message.

    Attributes:
        type: Always ``"image"``.
        data: Base64-encoded image data.
        mime_type: MIME type (e.g. ``"image/jpeg"``).
    """

    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType")


class ToolCall(BaseModelWithAliases):
    """An LLM request to invoke a tool.

    Attributes:
        type: Always ``"toolCall"``.
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: Parsed arguments matching the tool's JSON Schema.
        thought_signature: Opaque signature for reasoning context.
    """

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = Field(default=None, alias="thoughtSignature")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

UserContentItem = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]
"""Content block within a user message (text or image)."""

AssistantContentItem = Annotated[
    TextContent | ThinkingContent | ToolCall,
    Field(discriminator="type"),
]
"""Content block within an assistant message (text, thinking, or tool call)."""

ToolResultContentItem = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]
"""Content block within a tool result message (text or image)."""


class UserMessage(BaseModelWithAliases):
    """A message from the user.

    Attributes:
        role: Always ``"user"``.
        content: Plain text string or a list of content blocks.
        timestamp: Unix timestamp in milliseconds.
    """

    role: Literal["user"] = "user"
    content: str | list[UserContentItem]
    timestamp: int


class AssistantMessage(BaseModelWithAliases):
    """A response from the LLM assistant.

    All fields have defaults to support incremental construction during
    streaming. The message is progressively built up as events arrive.

    Attributes:
        role: Always ``"assistant"``.
        content: List of content blocks (text, thinking, tool calls).
        api: API identifier (e.g. ``"openai-responses"``).
        provider: Provider name (e.g. ``"openai"``).
        model: Model identifier used for this response.
        response_id: Provider-assigned response ID.
        usage: Token usage and cost breakdown.
        stop_reason: Why the response ended.
        error_message: Error description if ``stop_reason`` is ``"error"``.
        timestamp: Unix timestamp in milliseconds.
    """

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContentItem] = Field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_id: str | None = Field(default=None, alias="responseId")
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = Field(default="stop", alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    timestamp: int = 0


class ToolResultMessage(BaseModelWithAliases):
    """The result of executing a tool, sent back to the LLM.

    Attributes:
        role: Always ``"toolResult"``.
        tool_call_id: ID of the ``ToolCall`` this result responds to.
        tool_name: Name of the tool that was executed.
        content: Result content blocks (text or images).
        details: Arbitrary metadata about the execution.
        is_error: Whether the tool execution failed.
        timestamp: Unix timestamp in milliseconds.
    """

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[ToolResultContentItem] = Field(default_factory=list)
    details: Any | None = None
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int = 0


Message = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
"""Discriminated union of all message types (user, assistant, toolResult)."""

# ---------------------------------------------------------------------------
# Tool, Model, Context
# ---------------------------------------------------------------------------


class Tool(BaseModelWithAliases):
    """A tool definition that can be passed to an LLM.

    Tools enable LLMs to interact with external systems by defining a name,
    description, and JSON Schema for parameters.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description the LLM uses to decide
            when to call this tool.
        parameters: JSON Schema object describing the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_pydantic(
        cls, name: str, description: str, model_class: type[BaseModel]
    ) -> Tool:
        """Create a Tool from a Pydantic model class.

        The model's JSON schema is used as the tool's parameter schema.

        Args:
            name: Tool name.
            description: Tool description for the LLM.
            model_class: Pydantic model whose schema defines the parameters.

        Returns:
            A new Tool instance.
        """
        return cls(
            name=name,
            description=description,
            parameters=model_class.model_json_schema(),
        )


class Model(BaseModelWithAliases):
    """An LLM model descriptor.

    Constructed via ``get_model()`` or ``fetch_models()``. Contains all
    metadata needed to make API calls: endpoint, pricing, capabilities.

    Attributes:
        id: Model identifier (e.g. ``"gpt-4o"``).
        name: Display name.
        api: API backend identifier (e.g. ``"openai-responses"``).
        provider: Provider name (e.g. ``"openai"``).
        base_url: API base URL.
        reasoning: Whether the model supports extended thinking.
        input_types: Supported input modalities (e.g. ``["text", "image"]``).
        cost: Pricing per million tokens, or ``None`` if unknown.
        context_window: Maximum context window size in tokens.
        max_tokens: Maximum output tokens per request.
        headers: Custom HTTP headers to include in API requests.
    """

    id: str
    name: str
    api: str
    provider: str
    base_url: str = Field(default="https://api.openai.com/v1", alias="baseUrl")
    reasoning: bool = False
    input_types: list[str] = Field(
        default_factory=lambda: ["text"], alias="input"
    )
    cost: ModelCost | None = None
    context_window: int = Field(default=128000, alias="contextWindow")
    max_tokens: int = Field(default=16384, alias="maxTokens")
    headers: dict[str, str] | None = None


class Context(BaseModelWithAliases):
    """Conversation context passed to the LLM.

    A serializable snapshot of the current conversation state including
    system instructions, message history, and available tools.

    Attributes:
        system_prompt: System-level instructions for the LLM.
        messages: Conversation history.
        tools: Tools available to the LLM.
    """

    system_prompt: str = Field(default="", alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class StreamOptions:
    """Options for provider-level streaming calls.

    Attributes:
        temperature: Sampling temperature (``None`` for provider default).
        max_tokens: Maximum output tokens.
        api_key: API key override (falls back to environment variable).
        cancel_event: Event to signal request cancellation.
        cache_retention: Prompt caching preference.
        session_id: Session ID for prompt caching.
        on_payload: Hook called with raw API params before the request.
        headers: Additional HTTP headers.
        max_retry_delay_ms: Cap on retry backoff in milliseconds.
        metadata: Arbitrary metadata passed to the provider.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cancel_event: asyncio.Event | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    on_payload: Callable[[Any, Model], Any] | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    """Options for the simplified streaming API (``stream_simple`` / ``complete_simple``).

    Extends ``StreamOptions`` with reasoning support.

    Attributes:
        reasoning: Extended thinking level (``None`` to disable).
        thinking_budgets: Custom token budgets per thinking level.
    """

    reasoning: ThinkingLevel | None = None
    thinking_budgets: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StartEvent:
    """Emitted when streaming begins."""

    partial: AssistantMessage
    type: str = field(default="start", init=False)


@dataclass(slots=True)
class TextStartEvent:
    """Emitted when a new text content block begins streaming."""

    content_index: int
    partial: AssistantMessage
    type: str = field(default="text_start", init=False)


@dataclass(slots=True)
class TextDeltaEvent:
    """Emitted for each incremental chunk of text content."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = field(default="text_delta", init=False)


@dataclass(slots=True)
class TextEndEvent:
    """Emitted when a text content block finishes streaming."""

    content_index: int
    content: str
    partial: AssistantMessage
    type: str = field(default="text_end", init=False)


@dataclass(slots=True)
class ThinkingStartEvent:
    """Emitted when a thinking/reasoning block begins streaming."""

    content_index: int
    partial: AssistantMessage
    type: str = field(default="thinking_start", init=False)


@dataclass(slots=True)
class ThinkingDeltaEvent:
    """Emitted for each incremental chunk of thinking content."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = field(default="thinking_delta", init=False)


@dataclass(slots=True)
class ThinkingEndEvent:
    """Emitted when a thinking/reasoning block finishes streaming."""

    content_index: int
    content: str
    partial: AssistantMessage
    type: str = field(default="thinking_end", init=False)


@dataclass(slots=True)
class ToolCallStartEvent:
    """Emitted when a tool call begins streaming."""

    content_index: int
    partial: AssistantMessage
    type: str = field(default="toolcall_start", init=False)


@dataclass(slots=True)
class ToolCallDeltaEvent:
    """Emitted for each incremental chunk of tool call arguments."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = field(default="toolcall_delta", init=False)


@dataclass(slots=True)
class ToolCallEndEvent:
    """Emitted when a tool call finishes streaming and arguments are complete."""

    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: str = field(default="toolcall_end", init=False)


@dataclass(slots=True)
class DoneEvent:
    """Emitted when streaming completes successfully."""

    reason: StopReason
    message: AssistantMessage
    type: str = field(default="done", init=False)


@dataclass(slots=True)
class ErrorEvent:
    """Emitted when streaming ends due to an error or cancellation."""

    reason: StopReason
    error: AssistantMessage
    type: str = field(default="error", init=False)


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
"""Union of all streaming event types."""
