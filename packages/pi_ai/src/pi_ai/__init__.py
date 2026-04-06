"""pi_ai — Unified LLM provider abstraction."""

from pi_ai.api_registry import (
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)
from pi_ai.env_api_keys import get_env_api_key
from pi_ai.models import (
    calculate_cost,
    fetch_models,
    get_model,
    models_are_equal,
    supports_xhigh,
)
from pi_ai.providers import (
    OpenAIResponsesOptions,
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_reasoning,
    register_builtin_providers,
    reset_api_providers,
    stream_openai_responses,
    stream_simple_openai_responses,
)
from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
    register_faux_provider,
)
from pi_ai.stream import (
    complete,
    complete_simple,
    stream,
    stream_simple,
)
from pi_ai.types import (
    AssistantContentItem,
    AssistantMessage,
    AssistantMessageEvent,
    BaseModelWithAliases,
    CacheRetention,
    Context,
    CostBreakdown,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultContentItem,
    ToolResultMessage,
    Usage,
    UserContentItem,
    UserMessage,
)
from pi_ai.utils import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
    is_context_overflow,
    parse_streaming_json,
    sanitize_surrogates,
    short_hash,
    validate_tool_arguments,
    validate_tool_call,
)

__all__ = [
    # Base
    "BaseModelWithAliases",
    # Type aliases
    "StopReason",
    "ThinkingLevel",
    "CacheRetention",
    # Cost / Usage
    "CostBreakdown",
    "Usage",
    "ModelCost",
    # Content blocks
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    # Content unions
    "UserContentItem",
    "AssistantContentItem",
    "ToolResultContentItem",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    # Domain
    "Tool",
    "Model",
    "Context",
    # Options
    "StreamOptions",
    "SimpleStreamOptions",
    # Events
    "StartEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ThinkingStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "DoneEvent",
    "ErrorEvent",
    "AssistantMessageEvent",
    # Utils
    "EventStream",
    "AssistantMessageEventStream",
    "create_assistant_message_event_stream",
    "parse_streaming_json",
    "validate_tool_call",
    "validate_tool_arguments",
    "is_context_overflow",
    "short_hash",
    "sanitize_surrogates",
    # Env
    "get_env_api_key",
    # Registry
    "register_api_provider",
    "get_api_provider",
    "get_api_providers",
    "unregister_api_providers",
    "clear_api_providers",
    # Models
    "get_model",
    "fetch_models",
    "calculate_cost",
    "supports_xhigh",
    "models_are_equal",
    # Provider helpers
    "build_base_options",
    "clamp_reasoning",
    "adjust_max_tokens_for_thinking",
    "register_builtin_providers",
    "reset_api_providers",
    # OpenAI Responses provider
    "OpenAIResponsesOptions",
    "stream_openai_responses",
    "stream_simple_openai_responses",
    # Top-level streaming API
    "stream",
    "complete",
    "stream_simple",
    "complete_simple",
    # Faux provider (testing)
    "register_faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "faux_assistant_message",
]
