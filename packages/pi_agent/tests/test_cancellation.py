"""Tests for CancellationToken."""

import asyncio

from pi_agent import CancellationToken


def test_initial_state():
    token = CancellationToken()
    assert token.is_cancelled is False


def test_cancel():
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True


def test_cancel_idempotent():
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled is True


async def test_wait_resolves_on_cancel():
    token = CancellationToken()

    async def cancel_soon():
        await asyncio.sleep(0.01)
        token.cancel()

    task = asyncio.create_task(cancel_soon())
    await asyncio.wait_for(token.wait(), timeout=1.0)
    await task
    assert token.is_cancelled is True
