"""Shared fixtures for pi_llm_agent unit tests."""

import asyncio
import time

from pi_llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    TextContent,
    Usage,
)
from pi_llm.utils.event_stream import AssistantMessageEventStream


def make_mock_model(model_id: str = "mock") -> Model:
    """Create a mock Model for unit tests."""
    return Model(id=model_id, name=model_id, api="openai-responses", provider="openai")


def make_assistant_message(text: str = "ok", stop_reason: str = "stop") -> AssistantMessage:
    """Create an AssistantMessage for unit tests."""
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-responses",
        provider="openai",
        model="mock",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


def make_done_stream_fn(message: AssistantMessage | None = None):
    """Stream function that immediately pushes a DoneEvent."""
    msg = message or make_assistant_message()

    def stream_fn(*_args, **_kwargs):
        s = AssistantMessageEventStream()

        async def _push():
            await asyncio.sleep(0)
            s.push(DoneEvent(reason=msg.stop_reason, message=msg))
            s.end()

        asyncio.get_running_loop().create_task(_push())
        return s

    return stream_fn
