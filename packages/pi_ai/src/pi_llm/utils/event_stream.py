"""Push-based async iterable event stream.

Provides ``EventStream``, a generic producer/consumer bridge for streaming
LLM responses as typed events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar

from pi_llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
)

T = TypeVar("T")
R = TypeVar("R")

_SENTINEL = object()


class EventStream(Generic[T, R]):
    """Push-based async iterable stream with a final result.

    The producer calls ``push(event)`` to emit events and ``end()`` to close
    the stream. The consumer uses ``async for event in stream`` to iterate
    over events and ``await stream.result()`` to get the final result.

    Args:
        is_complete: Predicate that returns ``True`` for the terminal event.
        extract_result: Extracts the final result from the terminal event.
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._queue: asyncio.Queue[T | object] = asyncio.Queue()
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._done = False
        self._final_result: R | None = None
        self._result_set = False
        self._result_event = asyncio.Event()

    def push(self, event: T) -> None:
        """Push an event into the stream. Silently drops events after ``end()``."""
        if self._done:
            return
        if self._is_complete(event):
            self._done = True
            self._final_result = self._extract_result(event)
            self._result_set = True
            self._result_event.set()
        self._queue.put_nowait(event)

    def end(self, result: R | None = None) -> None:
        """Signal end of stream. Sets result if not already set."""
        self._done = True
        if result is not None and not self._result_set:
            self._final_result = result
            self._result_set = True
            self._result_event.set()
        self._queue.put_nowait(_SENTINEL)

    async def __aiter__(self) -> AsyncIterator[T]:
        """Async iterate over events until sentinel."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item  # type: ignore[misc]

    async def result(self) -> R:
        """Await the final result. Resolves on completion event or end(result)."""
        await self._result_event.wait()
        return self._final_result  # type: ignore[return-value]


def _extract_assistant_result(event: AssistantMessageEvent) -> AssistantMessage:
    if isinstance(event, DoneEvent):
        return event.message
    if isinstance(event, ErrorEvent):
        return event.error
    raise RuntimeError(f"Unexpected event type for final result: {type(event)}")


class AssistantMessageEventStream(
    EventStream[AssistantMessageEvent, AssistantMessage]
):
    """Event stream specialized for assistant message streaming."""

    def __init__(self) -> None:
        super().__init__(
            is_complete=lambda e: e.type in ("done", "error"),
            extract_result=_extract_assistant_result,
        )


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for AssistantMessageEventStream."""
    return AssistantMessageEventStream()
