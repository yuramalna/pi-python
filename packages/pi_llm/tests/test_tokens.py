"""Tests for token usage, cost calculation, and model functions."""

import pytest

from pi_llm import (
    CostBreakdown,
    Model,
    ModelCost,
    Usage,
    calculate_cost,
    get_model,
    models_are_equal,
    supports_xhigh,
)
from pi_llm.model_pricing import KNOWN_METADATA, KNOWN_PRICING


class TestCalculateCost:
    def test_calculates_cost_from_usage(self):
        model = get_model("openai", "gpt-4o")
        usage = Usage(input=1000, output=500, cache_read=200, total_tokens=1700)
        calculate_cost(model, usage)

        # gpt-4o: $2.5/M input, $10/M output, $1.25/M cache_read
        assert usage.cost.input == pytest.approx(1000 * 2.5 / 1_000_000)
        assert usage.cost.output == pytest.approx(500 * 10.0 / 1_000_000)
        assert usage.cost.cache_read == pytest.approx(200 * 1.25 / 1_000_000)
        assert usage.cost.total == pytest.approx(
            usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
        )

    def test_zero_cost_for_unknown_model(self):
        model = get_model("openai", "unknown-model-xyz")
        usage = Usage(input=1000, output=500, total_tokens=1500)
        calculate_cost(model, usage)

        assert usage.cost.total == 0.0

    def test_cost_with_cache_write(self):
        model = get_model("openai", "gpt-4o")
        usage = Usage(input=100, output=100, cache_write=500, total_tokens=700)
        calculate_cost(model, usage)

        # cache_write for gpt-4o is 0
        assert usage.cost.cache_write == 0.0

    def test_mutates_usage_in_place(self):
        model = get_model("openai", "o3")
        usage = Usage(input=5000, output=2000, total_tokens=7000)
        calculate_cost(model, usage)

        # o3: $10/M input (now $2/M), $40/M output (now $8/M)
        assert usage.cost.input > 0
        assert usage.cost.output > 0


class TestSupportsXhigh:
    def test_gpt_5_4_supports_xhigh(self):
        model = get_model("openai", "gpt-5.4")
        assert supports_xhigh(model) is True

    def test_gpt_5_2_supports_xhigh(self):
        model = Model(
            id="gpt-5.2", name="GPT-5.2",
            api="openai-responses", provider="openai",
        )
        assert supports_xhigh(model) is True

    def test_gpt_5_3_supports_xhigh(self):
        model = Model(
            id="gpt-5.3-codex", name="GPT-5.3 Codex",
            api="openai-responses", provider="openai",
        )
        assert supports_xhigh(model) is True

    def test_opus_4_6_supports_xhigh(self):
        model = Model(
            id="claude-opus-4-6", name="Claude Opus 4.6",
            api="anthropic-messages", provider="anthropic",
        )
        assert supports_xhigh(model) is True

    def test_opus_4_6_alt_format_supports_xhigh(self):
        model = Model(
            id="anthropic/claude-opus-4.6", name="Claude Opus 4.6",
            api="openai-completions", provider="openrouter",
        )
        assert supports_xhigh(model) is True

    def test_gpt_4o_does_not_support_xhigh(self):
        model = get_model("openai", "gpt-4o")
        assert supports_xhigh(model) is False

    def test_o3_does_not_support_xhigh(self):
        model = get_model("openai", "o3")
        assert supports_xhigh(model) is False

    def test_sonnet_does_not_support_xhigh(self):
        model = Model(
            id="claude-sonnet-4-5", name="Claude Sonnet 4.5",
            api="anthropic-messages", provider="anthropic",
        )
        assert supports_xhigh(model) is False


class TestModelsAreEqual:
    def test_same_model(self):
        a = get_model("openai", "gpt-4o")
        b = get_model("openai", "gpt-4o")
        assert models_are_equal(a, b) is True

    def test_different_model(self):
        a = get_model("openai", "gpt-4o")
        b = get_model("openai", "o3")
        assert models_are_equal(a, b) is False

    def test_none_values(self):
        assert models_are_equal(None, None) is False
        assert models_are_equal(get_model("openai", "gpt-4o"), None) is False
        assert models_are_equal(None, get_model("openai", "gpt-4o")) is False


class TestModelCatalog:
    def test_known_models_have_both_pricing_and_metadata(self):
        """Every model in pricing should also have metadata and vice versa."""
        pricing_only = set(KNOWN_PRICING) - set(KNOWN_METADATA)
        metadata_only = set(KNOWN_METADATA) - set(KNOWN_PRICING)
        assert pricing_only == set(), f"Models in pricing but not metadata: {pricing_only}"
        assert metadata_only == set(), f"Models in metadata but not pricing: {metadata_only}"

    def test_reasoning_models_have_correct_flag(self):
        reasoning_ids = {"o1", "o1-pro", "o3", "o3-mini", "o3-pro", "o4-mini", "gpt-5", "gpt-5.4"}
        for model_id in reasoning_ids:
            meta = KNOWN_METADATA.get(model_id)
            assert meta is not None, f"{model_id} not in KNOWN_METADATA"
            assert meta["reasoning"] is True, f"{model_id} should be reasoning=True"

    def test_non_reasoning_models_have_correct_flag(self):
        non_reasoning_ids = {"gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1"}
        for model_id in non_reasoning_ids:
            meta = KNOWN_METADATA.get(model_id)
            assert meta is not None, f"{model_id} not in KNOWN_METADATA"
            assert meta["reasoning"] is False, f"{model_id} should be reasoning=False"

    def test_get_model_returns_pricing_for_known_models(self):
        model = get_model("openai", "o3-pro")
        assert model.cost is not None
        assert model.cost.input == 20.0
        assert model.cost.output == 80.0

    def test_get_model_returns_metadata_for_known_models(self):
        model = get_model("openai", "gpt-5.4-pro")
        assert model.context_window == 1050000
        assert model.max_tokens == 128000
        assert model.reasoning is True

    def test_get_model_returns_defaults_for_unknown(self):
        model = get_model("openai", "totally-unknown-model")
        assert model.cost is None
        assert model.context_window == 128000
        assert model.max_tokens == 16384
        assert model.reasoning is False

    def test_catalog_has_at_least_40_models(self):
        assert len(KNOWN_PRICING) >= 40
        assert len(KNOWN_METADATA) >= 40
