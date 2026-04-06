"""Tests for context overflow detection across provider error patterns."""

import pytest

from pi_ai import AssistantMessage, Usage, is_context_overflow


def _error_msg(error_message: str) -> AssistantMessage:
    """Create an error AssistantMessage with the given error text."""
    return AssistantMessage(
        stop_reason="error",
        error_message=error_message,
    )


def _success_msg(input_tokens: int = 100, cache_read: int = 0) -> AssistantMessage:
    """Create a successful AssistantMessage with given usage."""
    return AssistantMessage(
        stop_reason="stop",
        usage=Usage(input=input_tokens, cache_read=cache_read, output=50, total_tokens=input_tokens + cache_read + 50),
    )


class TestOverflowPatterns:
    """Test that each provider's overflow error message is detected."""

    def test_anthropic_prompt_too_long(self):
        assert is_context_overflow(_error_msg("prompt is too long: 150000 tokens > 128000 maximum"))

    def test_anthropic_request_too_large(self):
        assert is_context_overflow(_error_msg("request_too_large"))

    def test_amazon_bedrock(self):
        assert is_context_overflow(_error_msg("input is too long for requested model"))

    def test_openai_responses(self):
        assert is_context_overflow(_error_msg("This request exceeds the context window of 128000 tokens"))

    def test_google_gemini(self):
        assert is_context_overflow(_error_msg("The input token count (200000) exceeds the maximum number of tokens"))

    def test_xai_grok(self):
        assert is_context_overflow(_error_msg("maximum prompt length is 131072"))

    def test_groq(self):
        assert is_context_overflow(_error_msg("Please reduce the length of the messages"))

    def test_openrouter(self):
        assert is_context_overflow(_error_msg("maximum context length is 128000 tokens"))

    def test_github_copilot(self):
        assert is_context_overflow(_error_msg("Total token count exceeds the limit of 128000"))

    def test_llama_cpp(self):
        assert is_context_overflow(_error_msg("exceeds the available context size"))

    def test_lm_studio(self):
        assert is_context_overflow(_error_msg("Input is greater than the context length of the model"))

    def test_minimax(self):
        assert is_context_overflow(_error_msg("context window exceeds limit"))

    def test_kimi_coding(self):
        assert is_context_overflow(_error_msg("exceeded model token limit"))

    def test_mistral(self):
        assert is_context_overflow(_error_msg("Request is too large for model with 32000 maximum context length"))

    def test_zai(self):
        assert is_context_overflow(_error_msg("model_context_window_exceeded"))

    def test_ollama_prompt_too_long(self):
        assert is_context_overflow(_error_msg("prompt too long; exceeded max context length by 100918 tokens"))

    def test_ollama_exceeded_context(self):
        assert is_context_overflow(_error_msg("prompt too long; exceeded context length by 500 tokens"))

    def test_cerebras_400(self):
        assert is_context_overflow(_error_msg("400 (no body)"))

    def test_cerebras_413(self):
        assert is_context_overflow(_error_msg("413 (no body)"))

    def test_generic_context_length_exceeded(self):
        assert is_context_overflow(_error_msg("context_length_exceeded"))

    def test_generic_context_length_exceeded_spaces(self):
        assert is_context_overflow(_error_msg("context length exceeded"))

    def test_generic_too_many_tokens(self):
        assert is_context_overflow(_error_msg("too many tokens in the input"))

    def test_generic_token_limit_exceeded(self):
        assert is_context_overflow(_error_msg("token limit exceeded"))


class TestNonOverflowPatterns:
    """Test that non-overflow errors are NOT detected as overflow."""

    def test_generic_error(self):
        assert not is_context_overflow(_error_msg("Internal server error"))

    def test_rate_limit(self):
        assert not is_context_overflow(_error_msg("Rate limit exceeded, please retry after 30 seconds."))

    def test_too_many_requests(self):
        assert not is_context_overflow(_error_msg("Too many requests. Please slow down."))

    def test_bedrock_throttling(self):
        assert not is_context_overflow(
            _error_msg("Throttling error: Too many tokens, please wait before trying again.")
        )

    def test_bedrock_service_unavailable(self):
        assert not is_context_overflow(
            _error_msg("Service unavailable: The service is temporarily unavailable.")
        )

    def test_ollama_non_overflow(self):
        assert not is_context_overflow(_error_msg("model runner crashed unexpectedly"))

    def test_no_error_message(self):
        assert not is_context_overflow(AssistantMessage(stop_reason="error"))

    def test_normal_stop(self):
        assert not is_context_overflow(AssistantMessage(stop_reason="stop"))

    def test_length_stop(self):
        assert not is_context_overflow(AssistantMessage(stop_reason="length"))

    def test_tool_use_stop(self):
        assert not is_context_overflow(AssistantMessage(stop_reason="toolUse"))


class TestSilentOverflow:
    """Test silent overflow detection via usage vs context window."""

    def test_detects_silent_overflow(self):
        msg = _success_msg(input_tokens=150000)
        assert is_context_overflow(msg, context_window=128000)

    def test_detects_silent_overflow_with_cache(self):
        msg = _success_msg(input_tokens=100000, cache_read=50000)
        assert is_context_overflow(msg, context_window=128000)

    def test_no_overflow_when_within_window(self):
        msg = _success_msg(input_tokens=100000)
        assert not is_context_overflow(msg, context_window=128000)

    def test_no_overflow_without_context_window(self):
        msg = _success_msg(input_tokens=999999)
        assert not is_context_overflow(msg)

    def test_no_silent_overflow_on_error(self):
        # Error messages should not trigger silent overflow check
        msg = AssistantMessage(
            stop_reason="error",
            error_message="some random error",
            usage=Usage(input=999999, total_tokens=999999),
        )
        assert not is_context_overflow(msg, context_window=128000)
