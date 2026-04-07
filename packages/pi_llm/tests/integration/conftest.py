"""Integration test configuration for pi_llm.

All tests in this directory require OPENAI_API_KEY.
Run: export $(cat ../../.env | xargs) && pytest tests/integration/ -v
"""

import os

import pytest

from pi_llm.providers.register_builtins import register_builtin_providers


def pytest_collection_modifyitems(config, items):
    """Skip all integration tests if OPENAI_API_KEY is not set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def openai_api_key():
    """Return the OPENAI_API_KEY or skip."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="session", autouse=True)
def _register_providers():
    """Register builtin providers once for the entire integration test session."""
    register_builtin_providers()
