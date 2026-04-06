"""Shared fixtures for pi_ai unit tests."""

import pytest

from pi_ai.api_registry import clear_api_providers
from pi_ai.utils.event_stream import AssistantMessageEventStream


@pytest.fixture()
def clean_registry():
    """Ensure a clean provider registry (explicit opt-in)."""
    clear_api_providers()
    yield
    clear_api_providers()


def mock_stream_fn(model, context, options=None):
    """Reusable mock stream function returning an empty AssistantMessageEventStream."""
    return AssistantMessageEventStream()


def mock_stream_simple_fn(model, context, options=None):
    """Reusable mock stream_simple function."""
    return AssistantMessageEventStream()
