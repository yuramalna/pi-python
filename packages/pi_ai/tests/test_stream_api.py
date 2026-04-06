"""Tests for the top-level stream API."""

import pytest

from pi_ai.api_registry import clear_api_providers, register_api_provider
from pi_ai.stream import _resolve_api_provider, stream, stream_simple
from pi_ai.types import Context, Model, UserMessage
from pi_ai.utils.event_stream import AssistantMessageEventStream


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_api_providers()
    yield
    clear_api_providers()


def _model(api="test-api"):
    return Model(id="test-model", name="test-model", api=api, provider="test")


def _ctx():
    return Context(messages=[UserMessage(content="Hi", timestamp=0)])


def _mock_stream(model, context, options=None):
    s = AssistantMessageEventStream()
    return s


def _mock_stream_simple(model, context, options=None):
    s = AssistantMessageEventStream()
    return s


# =============================================================================
# _resolve_api_provider
# =============================================================================


def test_resolve_provider_raises_for_unknown():
    with pytest.raises(ValueError, match="No API provider registered"):
        _resolve_api_provider("nonexistent-api")


def test_resolve_provider_succeeds():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    provider = _resolve_api_provider("test-api")
    assert provider.api == "test-api"


# =============================================================================
# stream / stream_simple
# =============================================================================


def test_stream_resolves_provider():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    result = stream(_model(), _ctx())
    assert isinstance(result, AssistantMessageEventStream)


def test_stream_raises_for_unknown_api():
    with pytest.raises(ValueError, match="No API provider"):
        stream(_model(api="unknown"), _ctx())


def test_stream_simple_resolves_provider():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    result = stream_simple(_model(), _ctx())
    assert isinstance(result, AssistantMessageEventStream)
