"""Context window overflow detection across providers.

Detects when an LLM response indicates the input exceeded the model's
context window, covering 20+ provider-specific error patterns.
"""

from __future__ import annotations

import re

from pi_llm.types import AssistantMessage

# Regex patterns to detect context overflow errors from different providers.
_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),  # Anthropic token overflow
    re.compile(r"request_too_large", re.IGNORECASE),  # Anthropic HTTP 413
    re.compile(r"input is too long for requested model", re.IGNORECASE),  # Amazon Bedrock
    re.compile(r"exceeds the context window", re.IGNORECASE),  # OpenAI
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),  # Google Gemini
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),  # xAI (Grok)
    re.compile(r"reduce the length of the messages", re.IGNORECASE),  # Groq
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),  # OpenRouter
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),  # GitHub Copilot
    re.compile(r"exceeds the available context size", re.IGNORECASE),  # llama.cpp
    re.compile(r"greater than the context length", re.IGNORECASE),  # LM Studio
    re.compile(r"context window exceeds limit", re.IGNORECASE),  # MiniMax
    re.compile(r"exceeded model token limit", re.IGNORECASE),  # Kimi For Coding
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),  # Mistral
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),  # z.ai
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),  # Ollama
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"too many tokens", re.IGNORECASE),  # Generic fallback
    re.compile(r"token limit exceeded", re.IGNORECASE),  # Generic fallback
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),  # Cerebras
]

# Non-overflow patterns (e.g. rate limiting) that should NOT be treated as overflow.
_NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),  # AWS Bedrock
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


def is_context_overflow(
    message: AssistantMessage, context_window: int | None = None
) -> bool:
    """Check if an assistant message represents a context overflow error.

    Handles two cases:
    1. Error-based overflow: Most providers return stop_reason "error" with a
       specific error message pattern.
    2. Silent overflow: Some providers (e.g. z.ai) accept overflow silently.
       For these, check if usage.input exceeds the context window.
    """
    # Case 1: Check error message patterns
    if message.stop_reason == "error" and message.error_message:
        is_non_overflow = any(
            p.search(message.error_message) for p in _NON_OVERFLOW_PATTERNS
        )
        if not is_non_overflow and any(
            p.search(message.error_message) for p in _OVERFLOW_PATTERNS
        ):
            return True

    # Case 2: Silent overflow — successful but usage exceeds context
    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    return False
