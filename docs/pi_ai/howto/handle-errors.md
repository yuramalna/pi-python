# Handle Errors

This guide covers error handling in pi-llm: detecting errors in the event stream, handling context overflow, and using cancellation to abort requests.

## Overview

Errors in pi-llm surface through three mechanisms:

- **`ErrorEvent`** -- Emitted when streaming ends due to an error or cancellation.
- **`is_context_overflow()`** -- Utility to detect context window overflow errors.
- **`cancel_event`** -- An `asyncio.Event` passed via options to abort in-flight requests.

## ErrorEvent

When an error occurs during streaming, the event stream emits an `ErrorEvent` instead of a `DoneEvent`:

```python
from pi_llm import ErrorEvent

async for event in event_stream:
    if isinstance(event, ErrorEvent):
        print(f"Error: {event.error.error_message}")
        print(f"Reason: {event.reason}")  # "error" or "aborted"
```

## Context overflow detection

Content coming soon.

## Cancellation

Content coming soon.

## Next steps

- [Events](../concepts/events.md) -- The ErrorEvent type
- [Streaming](../concepts/streaming.md) -- Cancellation via options
