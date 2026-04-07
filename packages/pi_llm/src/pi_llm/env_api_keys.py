"""Provider to environment variable mapping.

Maps provider names to their conventional environment variable names
for API key resolution.
"""

import os

_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def get_env_api_key(provider: str) -> str | None:
    """Get API key for provider from environment variables."""
    env_var = _PROVIDER_ENV_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None
