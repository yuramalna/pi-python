"""Cooperative cancellation token — Python equivalent of AbortSignal.

Passed to tool execute() and hook callbacks so they can check for
and respond to cancellation requests from the agent loop.
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    """Lightweight wrapper around ``asyncio.Event`` for cooperative cancellation."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal cancellation (like ``AbortController.abort()``)."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Block until cancellation is signalled."""
        await self._event.wait()
