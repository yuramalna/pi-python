# Models & Providers

pi-ai uses a registry-based architecture to support multiple LLM providers through a unified interface. Models are descriptors that carry all the metadata needed to make API calls, and providers are the adapters that translate pi-ai's streaming protocol into provider-specific API calls.

## Key concepts

- **Model** -- A data object describing an LLM endpoint (id, provider, pricing, capabilities).
- **Provider** -- An API adapter registered in the global registry.
- **`get_model(provider, model_id)`** -- Look up a model from the built-in catalog.
- **`register_builtin_providers()`** -- Register the built-in provider adapters (OpenAI Responses, etc.).
- **`register_api_provider()`** -- Register a custom provider adapter.

## The Model class

The `Model` class carries:

| Field | Description |
|---|---|
| `id` | Model identifier (e.g., `"gpt-4o"`) |
| `name` | Display name |
| `api` | API backend identifier (e.g., `"openai-responses"`) |
| `provider` | Provider name (e.g., `"openai"`) |
| `base_url` | API base URL |
| `reasoning` | Whether the model supports extended thinking |
| `input_types` | Supported input modalities (`["text"]`, `["text", "image"]`) |
| `cost` | Pricing per million tokens (`ModelCost`) |
| `context_window` | Maximum context size in tokens |
| `max_tokens` | Maximum output tokens per request |

## Provider registry

Content coming soon.

## Custom providers

Content coming soon.

## Next steps

- [Streaming](streaming.md) -- Making LLM calls with models
- [Cost Tracking](cost-tracking.md) -- Calculating costs from Usage data
