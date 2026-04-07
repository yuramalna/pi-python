"""Tests for API registry, model catalog, cost calculation, and simple options."""

import pytest

from pi_llm import (
    Context,
    SimpleStreamOptions,
    Usage,
    adjust_max_tokens_for_thinking,
    build_base_options,
    calculate_cost,
    clamp_reasoning,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    get_model,
    models_are_equal,
    register_api_provider,
    register_builtin_providers,
    reset_api_providers,
    supports_xhigh,
    unregister_api_providers,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream

# -- Fixtures --


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry for every test."""
    clear_api_providers()
    yield
    clear_api_providers()


def _mock_stream(model, context, options=None):
    return AssistantMessageEventStream()


def _mock_stream_simple(model, context, options=None):
    return AssistantMessageEventStream()


# -- API Registry --


def test_register_and_get_provider():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    provider = get_api_provider("test-api")
    assert provider is not None
    assert provider.api == "test-api"


def test_get_provider_not_found():
    assert get_api_provider("nonexistent") is None


def test_get_all_providers():
    register_api_provider("api-a", stream=_mock_stream, stream_simple=_mock_stream_simple)
    register_api_provider("api-b", stream=_mock_stream, stream_simple=_mock_stream_simple)
    providers = get_api_providers()
    assert len(providers) == 2
    apis = {p.api for p in providers}
    assert apis == {"api-a", "api-b"}


def test_api_mismatch_raises():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    provider = get_api_provider("test-api")
    model = get_model("openai", "gpt-4o")  # api="openai-responses"
    with pytest.raises(ValueError, match="Mismatched api"):
        provider.stream(model, Context())


def test_api_mismatch_raises_simple():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    provider = get_api_provider("test-api")
    model = get_model("openai", "gpt-4o")
    with pytest.raises(ValueError, match="Mismatched api"):
        provider.stream_simple(model, Context())


def test_unregister_by_source_id():
    register_api_provider("api-a", stream=_mock_stream, stream_simple=_mock_stream_simple, source_id="src1")
    register_api_provider("api-b", stream=_mock_stream, stream_simple=_mock_stream_simple, source_id="src2")
    unregister_api_providers("src1")
    assert get_api_provider("api-a") is None
    assert get_api_provider("api-b") is not None


def test_clear_all():
    register_api_provider("api-a", stream=_mock_stream, stream_simple=_mock_stream_simple)
    register_api_provider("api-b", stream=_mock_stream, stream_simple=_mock_stream_simple)
    clear_api_providers()
    assert get_api_providers() == []


def test_register_overwrites():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple, source_id="v1")
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple, source_id="v2")
    assert len(get_api_providers()) == 1


# -- Model Construction --


def test_get_model_known():
    model = get_model("openai", "gpt-4o")
    assert model.id == "gpt-4o"
    assert model.provider == "openai"
    assert model.api == "openai-responses"
    assert model.cost is not None
    assert model.cost.input == 2.5
    assert model.reasoning is False
    assert model.context_window == 128000


def test_get_model_unknown():
    model = get_model("openai", "future-model-xyz")
    assert model.cost is None
    assert model.context_window == 128000
    assert model.max_tokens == 16384


def test_get_model_overrides():
    model = get_model("openai", "gpt-4o", api="custom-api", base_url="http://localhost:8080")
    assert model.api == "custom-api"
    assert model.base_url == "http://localhost:8080"


def test_get_model_reasoning():
    model = get_model("openai", "o3")
    assert model.reasoning is True
    assert model.context_window == 200000


# -- Cost Calculation --


def test_calculate_cost_basic():
    model = get_model("openai", "gpt-4o")
    usage = Usage(input=1000, output=500)
    calculate_cost(model, usage)
    assert usage.cost.input == pytest.approx((2.5 / 1_000_000) * 1000)
    assert usage.cost.output == pytest.approx((10.0 / 1_000_000) * 500)
    assert usage.cost.total > 0
    assert usage.cost.total == pytest.approx(
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


def test_calculate_cost_no_pricing():
    model = get_model("openai", "future-model")
    usage = Usage(input=1000, output=500)
    calculate_cost(model, usage)
    # cost should remain default zeros
    assert usage.cost.total == 0


def test_calculate_cost_with_cache():
    model = get_model("openai", "gpt-4o")
    usage = Usage(input=1000, output=500, cache_read=200)
    calculate_cost(model, usage)
    assert usage.cost.cache_read == pytest.approx((1.25 / 1_000_000) * 200)


# -- supports_xhigh --


def test_supports_xhigh_true():
    assert supports_xhigh(get_model("openai", "gpt-5.2")) is True
    assert supports_xhigh(get_model("openai", "gpt-5.3-turbo")) is True
    assert supports_xhigh(get_model("openai", "gpt-5.4")) is True
    assert supports_xhigh(get_model("anthropic", "claude-opus-4-6")) is True
    assert supports_xhigh(get_model("anthropic", "claude-opus-4.6")) is True


def test_supports_xhigh_false():
    assert supports_xhigh(get_model("openai", "gpt-4o")) is False
    assert supports_xhigh(get_model("openai", "o3")) is False


# -- models_are_equal --


def test_models_equal():
    a = get_model("openai", "gpt-4o")
    b = get_model("openai", "gpt-4o")
    assert models_are_equal(a, b) is True


def test_models_not_equal_id():
    a = get_model("openai", "gpt-4o")
    b = get_model("openai", "gpt-4o-mini")
    assert models_are_equal(a, b) is False


def test_models_not_equal_provider():
    a = get_model("openai", "gpt-4o")
    b = get_model("custom", "gpt-4o")
    assert models_are_equal(a, b) is False


def test_models_equal_none():
    model = get_model("openai", "gpt-4o")
    assert models_are_equal(None, model) is False
    assert models_are_equal(model, None) is False
    assert models_are_equal(None, None) is False


# -- build_base_options --


def test_build_base_options_defaults():
    model = get_model("openai", "gpt-4o")  # max_tokens=16384
    opts = build_base_options(model)
    assert opts.max_tokens == min(16384, 32000)
    assert opts.temperature is None
    assert opts.api_key is None


def test_build_base_options_with_options():
    model = get_model("openai", "gpt-4o")
    opts = build_base_options(model, SimpleStreamOptions(temperature=0.7, max_tokens=4096))
    assert opts.temperature == 0.7
    assert opts.max_tokens == 4096


def test_build_base_options_api_key_precedence():
    model = get_model("openai", "gpt-4o")
    opts = build_base_options(
        model, SimpleStreamOptions(api_key="from-options"), api_key="explicit"
    )
    assert opts.api_key == "explicit"


def test_build_base_options_api_key_from_options():
    model = get_model("openai", "gpt-4o")
    opts = build_base_options(model, SimpleStreamOptions(api_key="from-options"))
    assert opts.api_key == "from-options"


# -- clamp_reasoning --


def test_clamp_reasoning_xhigh():
    assert clamp_reasoning("xhigh") == "high"


def test_clamp_reasoning_passthrough():
    assert clamp_reasoning("medium") == "medium"
    assert clamp_reasoning("low") == "low"
    assert clamp_reasoning(None) is None


# -- adjust_max_tokens_for_thinking --


def test_adjust_max_tokens_basic():
    # base=16384, model_max=100000, level=medium → budget=8192
    max_tokens, budget = adjust_max_tokens_for_thinking(16384, 100000, "medium")
    assert budget == 8192
    assert max_tokens == 16384 + 8192


def test_adjust_max_tokens_clamped_to_model_max():
    # base=90000, model_max=100000, level=high → budget=16384
    # max_tokens = min(90000+16384, 100000) = 100000
    max_tokens, budget = adjust_max_tokens_for_thinking(90000, 100000, "high")
    assert max_tokens == 100000
    assert budget == 16384


def test_adjust_max_tokens_budget_clamped():
    # base=100, model_max=500, level=high → budget=16384
    # max_tokens = min(100+16384, 500) = 500
    # 500 <= 16384 → budget = max(0, 500 - 1024) = 0
    max_tokens, budget = adjust_max_tokens_for_thinking(100, 500, "high")
    assert max_tokens == 500
    assert budget == 0


def test_adjust_max_tokens_xhigh_clamped():
    # xhigh should be clamped to high
    _max_tokens, budget = adjust_max_tokens_for_thinking(16384, 100000, "xhigh")
    assert budget == 16384


def test_adjust_max_tokens_custom_budgets():
    max_tokens, budget = adjust_max_tokens_for_thinking(
        16384, 100000, "medium", custom_budgets={"medium": 4096}
    )
    assert budget == 4096
    assert max_tokens == 16384 + 4096


# -- Register builtins --


def test_register_builtin_registers_openai():
    register_builtin_providers()
    providers = get_api_providers()
    assert len(providers) == 1
    assert providers[0].api == "openai-responses"


def test_reset_api_providers():
    register_api_provider("test-api", stream=_mock_stream, stream_simple=_mock_stream_simple)
    assert len(get_api_providers()) == 1
    reset_api_providers()
    # After reset, only builtins remain (openai-responses from M4)
    providers = get_api_providers()
    assert len(providers) == 1
    assert providers[0].api == "openai-responses"
