"""Tests for EventStream and AssistantMessageEventStream."""

import asyncio

from pi_ai import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    EventStream,
    StartEvent,
)


async def test_push_and_iterate():
    stream = EventStream(
        is_complete=lambda e: e == "done", extract_result=lambda e: e
    )
    stream.push("a")
    stream.push("b")
    stream.push("done")
    stream.end()
    events = [e async for e in stream]
    assert events == ["a", "b", "done"]
    assert await stream.result() == "done"


async def test_push_after_end_is_noop():
    stream = EventStream(
        is_complete=lambda e: False, extract_result=lambda e: None
    )
    stream.push("a")
    stream.end(result="r")
    stream.push("b")  # dropped — stream is done
    events = [e async for e in stream]
    assert events == ["a"]
    assert await stream.result() == "r"


async def test_concurrent_producer_consumer():
    stream = EventStream(
        is_complete=lambda e: e == "done", extract_result=lambda e: e
    )

    async def producer():
        for i in range(5):
            stream.push(str(i))
            await asyncio.sleep(0.01)
        stream.push("done")
        stream.end()

    async def consumer():
        return [e async for e in stream]

    _, events = await asyncio.gather(producer(), consumer())
    assert events == ["0", "1", "2", "3", "4", "done"]


async def test_result_resolves_on_completion_event():
    stream = EventStream(
        is_complete=lambda e: e == "fin", extract_result=lambda e: "result_val"
    )
    stream.push("x")
    stream.push("fin")
    # result is set immediately by push, no need for end()
    assert await stream.result() == "result_val"


async def test_end_with_explicit_result():
    stream = EventStream(
        is_complete=lambda e: False, extract_result=lambda e: None
    )
    stream.push("a")
    stream.end(result="explicit")
    assert await stream.result() == "explicit"


async def test_end_does_not_override_completion_result():
    stream = EventStream(
        is_complete=lambda e: e == "done",
        extract_result=lambda e: "from_event",
    )
    stream.push("done")  # sets result to "from_event"
    stream.end(result="from_end")  # should NOT override
    assert await stream.result() == "from_event"


async def test_assistant_stream_done():
    stream = AssistantMessageEventStream()
    msg = AssistantMessage(
        api="test", provider="test", model="test", timestamp=0
    )
    stream.push(StartEvent(partial=msg))
    stream.push(DoneEvent(reason="stop", message=msg))
    stream.end()
    events = [e async for e in stream]
    assert len(events) == 2
    assert events[0].type == "start"
    assert events[1].type == "done"
    assert await stream.result() is msg


async def test_assistant_stream_error():
    stream = AssistantMessageEventStream()
    msg = AssistantMessage(
        stop_reason="error", error_message="fail", timestamp=0
    )
    stream.push(ErrorEvent(reason="error", error=msg))
    stream.end()
    result = await stream.result()
    assert result.error_message == "fail"
