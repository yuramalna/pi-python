"""Tests for pi_ai core types — milestone 1 verification."""

from pydantic import BaseModel, TypeAdapter

from pi_ai import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    get_env_api_key,
)

# -- Construction --


def test_user_message_construction():
    msg = UserMessage(content="Hello", timestamp=1000)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp == 1000


def test_user_message_with_content_list():
    msg = UserMessage(
        content=[TextContent(text="Hi"), ImageContent(data="abc", mimeType="image/png")],
        timestamp=0,
    )
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextContent)
    assert isinstance(msg.content[1], ImageContent)


def test_assistant_message_defaults():
    msg = AssistantMessage(timestamp=0)
    assert msg.role == "assistant"
    assert msg.content == []
    assert msg.stop_reason == "stop"
    assert msg.usage.input == 0
    assert msg.usage.cost.total == 0.0


def test_tool_result_message_construction():
    msg = ToolResultMessage(
        tool_call_id="tc1", tool_name="search", is_error=False, timestamp=0
    )
    assert msg.role == "toolResult"
    assert msg.tool_call_id == "tc1"


# -- camelCase serialization --


def test_camelcase_serialization():
    msg = AssistantMessage(stop_reason="toolUse", timestamp=0)
    data = msg.model_dump(by_alias=True)
    assert "stopReason" in data
    assert data["stopReason"] == "toolUse"
    assert "responseId" in data
    assert "errorMessage" in data


def test_usage_camelcase():
    u = Usage(cache_read=10, total_tokens=100)
    data = u.model_dump(by_alias=True)
    assert "cacheRead" in data
    assert data["cacheRead"] == 10
    assert "totalTokens" in data
    assert data["totalTokens"] == 100


# -- camelCase input acceptance --


def test_camelcase_input():
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="search", isError=False, timestamp=0
    )
    assert msg.tool_call_id == "tc1"
    assert msg.tool_name == "search"
    assert msg.is_error is False


def test_model_camelcase_input():
    m = Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        baseUrl="https://api.openai.com/v1",
        contextWindow=128000,
        maxTokens=16384,
    )
    assert m.base_url == "https://api.openai.com/v1"
    assert m.context_window == 128000


# -- Discriminated union --


def test_message_discriminated_union():
    adapter = TypeAdapter(list[Message])
    messages = adapter.validate_python(
        [
            {"role": "user", "content": "Hi", "timestamp": 0},
            {
                "role": "assistant",
                "content": [],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
                "usage": {},
                "stopReason": "stop",
                "timestamp": 0,
            },
            {
                "role": "toolResult",
                "toolCallId": "1",
                "toolName": "t",
                "isError": False,
                "timestamp": 0,
            },
        ]
    )
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert isinstance(messages[2], ToolResultMessage)


# -- Tool.from_pydantic --


def test_tool_from_pydantic():
    class SearchParams(BaseModel):
        query: str
        limit: int = 10

    tool = Tool.from_pydantic("search", "Search the web", SearchParams)
    assert tool.name == "search"
    assert "properties" in tool.parameters
    assert "query" in tool.parameters["properties"]


# -- Mutable streaming pattern --


def test_assistant_message_mutable():
    msg = AssistantMessage(
        api="openai-responses", provider="openai", model="gpt-4o", timestamp=1000
    )
    msg.content.append(TextContent(text="Hello"))
    msg.stop_reason = "stop"
    msg.usage.input = 50
    assert len(msg.content) == 1
    assert msg.content[0].text == "Hello"
    assert msg.usage.input == 50


# -- Events --


def test_start_event():
    msg = AssistantMessage(timestamp=0)
    event = StartEvent(partial=msg)
    assert event.type == "start"
    assert event.partial is msg


def test_done_event():
    msg = AssistantMessage(timestamp=0)
    event = DoneEvent(reason="stop", message=msg)
    assert event.type == "done"
    assert event.reason == "stop"


def test_error_event():
    msg = AssistantMessage(stop_reason="error", error_message="fail", timestamp=0)
    event = ErrorEvent(reason="error", error=msg)
    assert event.type == "error"


# -- Options --


def test_stream_options_defaults():
    opts = StreamOptions()
    assert opts.temperature is None
    assert opts.api_key is None


def test_simple_stream_options():
    opts = SimpleStreamOptions(reasoning="high", temperature=0.7)
    assert opts.reasoning == "high"
    assert opts.temperature == 0.7


# -- Content blocks --


def test_text_content():
    tc = TextContent(text="hello")
    assert tc.type == "text"
    assert tc.text_signature is None


def test_thinking_content():
    tc = ThinkingContent(thinking="let me think...")
    assert tc.type == "thinking"
    assert tc.redacted is None


def test_tool_call():
    tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
    assert tc.type == "toolCall"
    assert tc.arguments["query"] == "test"


# -- Model --


def test_model_defaults():
    m = Model(id="gpt-4o", name="GPT-4o", api="openai-responses", provider="openai")
    assert m.base_url == "https://api.openai.com/v1"
    assert m.reasoning is False
    assert m.input_types == ["text"]
    assert m.context_window == 128000
    assert m.max_tokens == 16384
    assert m.cost is None


def test_model_with_cost():
    cost = ModelCost(input=2.5, output=10.0, cache_read=1.25)
    m = Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-responses",
        provider="openai",
        cost=cost,
    )
    assert m.cost.input == 2.5
    assert m.cost.cache_read == 1.25


# -- Env API keys --


def test_get_env_api_key_known(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_env_api_key("openai") == "sk-test"


def test_get_env_api_key_unknown():
    assert get_env_api_key("unknown-provider") is None
