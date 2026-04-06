"""Built-in provider registration.

Call ``register_builtin_providers()`` once at startup before using
the streaming API.
"""

from pi_ai.api_registry import clear_api_providers, register_api_provider


def register_builtin_providers() -> None:
    """Register built-in API providers.

    Currently registers the OpenAI Responses API provider. Must be called
    before ``stream_simple()`` or ``complete_simple()`` will work.
    """
    try:
        from pi_ai.providers.openai_responses import (
            stream_openai_responses,
            stream_simple_openai_responses,
        )

        register_api_provider(
            api="openai-responses",
            stream=stream_openai_responses,
            stream_simple=stream_simple_openai_responses,
        )
    except ImportError:
        pass  # Provider not yet implemented (M4)


def reset_api_providers() -> None:
    """Clear all providers and re-register builtins."""
    clear_api_providers()
    register_builtin_providers()
