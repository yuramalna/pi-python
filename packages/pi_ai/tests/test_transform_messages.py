"""Tests for cross-provider message transformation."""


from pi_ai.providers.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


def _model(*, id: str = "gpt-4o", provider: str = "openai", api: str = "openai-responses") -> Model:
    return Model(id=id, name=id, api=api, provider=provider)


def _assistant(
    content, *, provider="openai", api="openai-responses", model="gpt-4o", stop_reason="stop"
):
    return AssistantMessage(
        content=content, provider=provider, api=api, model=model,
        stop_reason=stop_reason, timestamp=0,
    )


# -- Pass 1 tests --


def test_user_messages_pass_through():
    user = UserMessage(content="hello", timestamp=0)
    result = transform_messages([user], _model())
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "hello"


def test_tool_result_id_normalized():
    """ToolResult IDs are updated when the map has an entry from tool call normalization."""
    tc = ToolCall(id="original_id", name="fn", arguments={})
    assistant = _assistant([tc], provider="other", api="other-api")
    tr = ToolResultMessage(tool_call_id="original_id", tool_name="fn", timestamp=0)

    def normalize(id, model, source):
        return "normalized_id"

    result = transform_messages([assistant, tr], _model(), normalize)
    # Assistant's tool call should be normalized
    assert result[0].content[0].id == "normalized_id"
    # ToolResult should pick up the normalized ID
    assert result[1].tool_call_id == "normalized_id"


def test_same_model_keeps_thinking_signature():
    thinking = ThinkingContent(thinking="deep thought", thinking_signature='{"v":1}')
    assistant = _assistant([thinking])
    result = transform_messages([assistant], _model())
    assert result[0].content[0].type == "thinking"
    assert result[0].content[0].thinking_signature == '{"v":1}'


def test_cross_model_drops_redacted_thinking():
    thinking = ThinkingContent(thinking="", redacted=True)
    assistant = _assistant([thinking], model="different-model")
    result = transform_messages([assistant], _model())
    assert len(result[0].content) == 0


def test_same_model_keeps_redacted_thinking():
    thinking = ThinkingContent(thinking="", redacted=True)
    assistant = _assistant([thinking])
    result = transform_messages([assistant], _model())
    assert len(result[0].content) == 1
    assert result[0].content[0].redacted is True


def test_cross_model_thinking_to_text():
    thinking = ThinkingContent(thinking="my reasoning")
    assistant = _assistant([thinking], model="different-model")
    result = transform_messages([assistant], _model())
    assert result[0].content[0].type == "text"
    assert result[0].content[0].text == "my reasoning"


def test_empty_thinking_skipped():
    thinking = ThinkingContent(thinking="   ")
    assistant = _assistant([thinking])
    result = transform_messages([assistant], _model())
    assert len(result[0].content) == 0


def test_cross_model_strips_thought_signature():
    tc = ToolCall(id="tc1", name="fn", arguments={}, thought_signature="sig")
    assistant = _assistant([tc], model="different-model")
    result = transform_messages([assistant], _model())
    assert result[0].content[0].thought_signature is None


def test_cross_model_strips_text_signature():
    text = TextContent(text="hello", text_signature="sig123")
    assistant = _assistant([text], model="different-model")
    result = transform_messages([assistant], _model())
    assert result[0].content[0].text == "hello"
    assert result[0].content[0].text_signature is None


# -- Pass 2 tests --


def test_orphaned_tool_calls_get_synthetic_results():
    tc = ToolCall(id="tc1", name="fn", arguments={})
    assistant = _assistant([tc])
    user = UserMessage(content="continue", timestamp=0)
    result = transform_messages([assistant, user], _model())
    # Should be: assistant, synthetic toolResult, user
    assert len(result) == 3
    assert result[1].role == "toolResult"
    assert result[1].tool_call_id == "tc1"
    assert result[1].is_error is True


def test_errored_assistant_skipped():
    tc = ToolCall(id="tc1", name="fn", arguments={})
    assistant = _assistant([tc], stop_reason="error")
    user = UserMessage(content="retry", timestamp=0)
    result = transform_messages([assistant, user], _model())
    # Errored assistant is skipped entirely
    assert len(result) == 1
    assert result[0].role == "user"


def test_aborted_assistant_skipped():
    assistant = _assistant([TextContent(text="partial")], stop_reason="aborted")
    result = transform_messages([assistant], _model())
    assert len(result) == 0


def test_existing_tool_results_not_duplicated():
    tc = ToolCall(id="tc1", name="fn", arguments={})
    assistant = _assistant([tc])
    tr = ToolResultMessage(tool_call_id="tc1", tool_name="fn", timestamp=0)
    user = UserMessage(content="ok", timestamp=0)
    result = transform_messages([assistant, tr, user], _model())
    # No synthetic result needed — real result exists
    assert len(result) == 3
    assert result[0].role == "assistant"
    assert result[1].role == "toolResult"
    assert result[2].role == "user"
