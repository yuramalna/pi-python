"""Tests for OpenAI Responses provider entry points."""

import os
from unittest.mock import patch

import pytest

from pi_ai.providers.openai_responses import (
    OpenAIResponsesOptions,
    _apply_service_tier_pricing,
    _build_params,
    _get_service_tier_cost_multiplier,
    _resolve_cache_retention,
    stream_simple_openai_responses,
)
from pi_ai.types import Context, Model, SimpleStreamOptions, Usage, UserMessage


def _model(*, id="gpt-4o", provider="openai", reasoning=False):
    return Model(
        id=id, name=id, api="openai-responses", provider=provider, reasoning=reasoning,
    )


def _ctx(system_prompt="", tools=None):
    return Context(
        system_prompt=system_prompt,
        messages=[UserMessage(content="Hello", timestamp=0)],
        tools=tools or [],
    )


# =============================================================================
# _build_params
# =============================================================================


def test_build_params_basic():
    params = _build_params(_model(), _ctx())
    assert params["model"] == "gpt-4o"
    assert params["stream"] is True
    assert params["store"] is False
    assert "input" in params
    assert isinstance(params["input"], list)


def test_build_params_with_options():
    opts = OpenAIResponsesOptions(max_tokens=4096, temperature=0.5)
    params = _build_params(_model(), _ctx(), opts)
    assert params["max_output_tokens"] == 4096
    assert params["temperature"] == 0.5


def test_build_params_reasoning_model_with_effort():
    model = _model(reasoning=True)
    opts = OpenAIResponsesOptions(reasoning_effort="high", reasoning_summary="auto")
    params = _build_params(model, _ctx(), opts)
    assert params["reasoning"]["effort"] == "high"
    assert params["reasoning"]["summary"] == "auto"
    assert "reasoning.encrypted_content" in params["include"]


def test_build_params_reasoning_model_no_effort():
    model = _model(reasoning=True)
    params = _build_params(model, _ctx())
    assert params["reasoning"]["effort"] == "none"


def test_build_params_non_reasoning_no_reasoning_key():
    params = _build_params(_model(), _ctx())
    assert "reasoning" not in params


def test_build_params_cache_retention_session():
    opts = OpenAIResponsesOptions(cache_retention="short", session_id="sess_123")
    params = _build_params(_model(), _ctx(), opts)
    assert params["prompt_cache_key"] == "sess_123"


def test_build_params_cache_retention_none():
    opts = OpenAIResponsesOptions(cache_retention="none", session_id="sess_123")
    params = _build_params(_model(), _ctx(), opts)
    assert "prompt_cache_key" not in params


# =============================================================================
# _resolve_cache_retention
# =============================================================================


def test_resolve_cache_retention_explicit():
    assert _resolve_cache_retention("long") == "long"
    assert _resolve_cache_retention("none") == "none"


def test_resolve_cache_retention_default():
    assert _resolve_cache_retention(None) == "short"


def test_resolve_cache_retention_env_var():
    with patch.dict(os.environ, {"PI_CACHE_RETENTION": "long"}):
        assert _resolve_cache_retention(None) == "long"


# =============================================================================
# Service tier pricing
# =============================================================================


def test_service_tier_cost_multiplier():
    assert _get_service_tier_cost_multiplier("flex") == 0.5
    assert _get_service_tier_cost_multiplier("priority") == 2.0
    assert _get_service_tier_cost_multiplier(None) == 1.0
    assert _get_service_tier_cost_multiplier("auto") == 1.0


def test_apply_service_tier_pricing_flex():
    usage = Usage(input=100, output=50)
    usage.cost.input = 1.0
    usage.cost.output = 2.0
    usage.cost.total = 3.0
    _apply_service_tier_pricing(usage, "flex")
    assert usage.cost.input == pytest.approx(0.5)
    assert usage.cost.output == pytest.approx(1.0)
    assert usage.cost.total == pytest.approx(1.5)


def test_apply_service_tier_pricing_default_noop():
    usage = Usage()
    usage.cost.input = 1.0
    usage.cost.total = 1.0
    _apply_service_tier_pricing(usage, None)
    assert usage.cost.input == pytest.approx(1.0)


# =============================================================================
# stream_simple validation
# =============================================================================


def test_stream_simple_validates_api_key():
    model = _model(provider="custom-provider")
    ctx = _ctx()
    with pytest.raises(ValueError, match="No API key"):
        stream_simple_openai_responses(model, ctx)


def test_stream_simple_validates_api_key_with_key():
    """With an API key provided, should not raise (returns stream)."""
    model = _model()
    ctx = _ctx()
    opts = SimpleStreamOptions(api_key="test-key")
    # This will try to create the async task, which requires a running loop.
    # We just verify it doesn't raise ValueError about missing API key.
    # The RuntimeError about no running loop is expected in sync test context.
    try:
        stream_simple_openai_responses(model, ctx, opts)
    except RuntimeError as e:
        assert "no running event loop" in str(e).lower() or "no current event loop" in str(e).lower()
