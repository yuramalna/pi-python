"""Known model pricing and metadata.

Sourced from the TypeScript model catalog (models.generated.ts).
All models listed here use the ``openai-responses`` API via the ``openai`` provider.
"""

from pi_ai.types import ModelCost

# $/million tokens
KNOWN_PRICING: dict[str, ModelCost] = {
    # --- GPT-4 family ---
    "gpt-4": ModelCost(input=30.0, output=60.0),
    "gpt-4-turbo": ModelCost(input=10.0, output=30.0),
    # --- GPT-4o family ---
    "gpt-4o": ModelCost(input=2.5, output=10.0, cache_read=1.25),
    "gpt-4o-2024-05-13": ModelCost(input=5.0, output=15.0),
    "gpt-4o-2024-08-06": ModelCost(input=2.5, output=10.0, cache_read=1.25),
    "gpt-4o-2024-11-20": ModelCost(input=2.5, output=10.0, cache_read=1.25),
    "gpt-4o-mini": ModelCost(input=0.15, output=0.6, cache_read=0.08),
    # --- GPT-4.1 family ---
    "gpt-4.1": ModelCost(input=2.0, output=8.0, cache_read=0.5),
    "gpt-4.1-mini": ModelCost(input=0.4, output=1.6, cache_read=0.1),
    "gpt-4.1-nano": ModelCost(input=0.1, output=0.4, cache_read=0.03),
    # --- o-series reasoning ---
    "o1": ModelCost(input=15.0, output=60.0, cache_read=7.5),
    "o1-pro": ModelCost(input=150.0, output=600.0),
    "o3": ModelCost(input=2.0, output=8.0, cache_read=0.5),
    "o3-deep-research": ModelCost(input=10.0, output=40.0, cache_read=2.5),
    "o3-mini": ModelCost(input=1.1, output=4.4, cache_read=0.55),
    "o3-pro": ModelCost(input=20.0, output=80.0),
    "o4-mini": ModelCost(input=1.1, output=4.4, cache_read=0.28),
    "o4-mini-deep-research": ModelCost(input=2.0, output=8.0, cache_read=0.5),
    # --- Codex ---
    "codex-mini-latest": ModelCost(input=1.5, output=6.0, cache_read=0.375),
    # --- GPT-5 family ---
    "gpt-5": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5-chat-latest": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5-codex": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5-mini": ModelCost(input=0.25, output=2.0, cache_read=0.025),
    "gpt-5-nano": ModelCost(input=0.05, output=0.4, cache_read=0.005),
    "gpt-5-pro": ModelCost(input=15.0, output=120.0),
    # --- GPT-5.1 family ---
    "gpt-5.1": ModelCost(input=1.25, output=10.0, cache_read=0.13),
    "gpt-5.1-chat-latest": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5.1-codex": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5.1-codex-max": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    "gpt-5.1-codex-mini": ModelCost(input=0.25, output=2.0, cache_read=0.025),
    # --- GPT-5.2 family ---
    "gpt-5.2": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.2-chat-latest": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.2-codex": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.2-pro": ModelCost(input=21.0, output=168.0),
    # --- GPT-5.3 family ---
    "gpt-5.3-chat-latest": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.3-codex": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    "gpt-5.3-codex-spark": ModelCost(input=1.75, output=14.0, cache_read=0.175),
    # --- GPT-5.4 family ---
    "gpt-5.4": ModelCost(input=2.5, output=15.0, cache_read=0.25),
    "gpt-5.4-2026-03-05": ModelCost(input=2.5, output=15.0, cache_read=0.25),
    "gpt-5.4-mini": ModelCost(input=0.75, output=4.5, cache_read=0.075),
    "gpt-5.4-nano": ModelCost(input=0.2, output=1.25, cache_read=0.02),
    "gpt-5.4-pro": ModelCost(input=30.0, output=180.0),
}

# Known model metadata (context window, max tokens, capabilities)
KNOWN_METADATA: dict[str, dict] = {
    # --- GPT-4 family ---
    "gpt-4": {"context_window": 8192, "max_tokens": 8192, "reasoning": False, "input": ["text"]},
    "gpt-4-turbo": {"context_window": 128000, "max_tokens": 4096, "reasoning": False, "input": ["text", "image"]},
    # --- GPT-4o family ---
    "gpt-4o": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    "gpt-4o-2024-05-13": {"context_window": 128000, "max_tokens": 4096, "reasoning": False, "input": ["text", "image"]},
    "gpt-4o-2024-08-06": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    "gpt-4o-2024-11-20": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    "gpt-4o-mini": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    # --- GPT-4.1 family ---
    "gpt-4.1": {"context_window": 1047576, "max_tokens": 32768, "reasoning": False, "input": ["text", "image"]},
    "gpt-4.1-mini": {"context_window": 1047576, "max_tokens": 32768, "reasoning": False, "input": ["text", "image"]},
    "gpt-4.1-nano": {"context_window": 1047576, "max_tokens": 32768, "reasoning": False, "input": ["text", "image"]},
    # --- o-series reasoning ---
    "o1": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o1-pro": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o3": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o3-deep-research": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o3-mini": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text"]},
    "o3-pro": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o4-mini": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    "o4-mini-deep-research": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text", "image"]},
    # --- Codex ---
    "codex-mini-latest": {"context_window": 200000, "max_tokens": 100000, "reasoning": True, "input": ["text"]},
    # --- GPT-5 family ---
    "gpt-5": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5-chat-latest": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    "gpt-5-codex": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5-mini": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5-nano": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5-pro": {"context_window": 400000, "max_tokens": 272000, "reasoning": True, "input": ["text", "image"]},
    # --- GPT-5.1 family ---
    "gpt-5.1": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.1-chat-latest": {"context_window": 128000, "max_tokens": 16384, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.1-codex": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.1-codex-max": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.1-codex-mini": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    # --- GPT-5.2 family ---
    "gpt-5.2": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.2-chat-latest": {"context_window": 128000, "max_tokens": 16384, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.2-codex": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.2-pro": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    # --- GPT-5.3 family ---
    "gpt-5.3-chat-latest": {"context_window": 128000, "max_tokens": 16384, "reasoning": False, "input": ["text", "image"]},
    "gpt-5.3-codex": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.3-codex-spark": {"context_window": 128000, "max_tokens": 32000, "reasoning": True, "input": ["text", "image"]},
    # --- GPT-5.4 family ---
    "gpt-5.4": {"context_window": 272000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.4-2026-03-05": {"context_window": 272000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.4-mini": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.4-nano": {"context_window": 400000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
    "gpt-5.4-pro": {"context_window": 1050000, "max_tokens": 128000, "reasoning": True, "input": ["text", "image"]},
}
