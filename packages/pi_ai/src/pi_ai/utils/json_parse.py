"""Streaming JSON parser for incomplete tool call arguments."""

from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json


def parse_streaming_json(partial: str) -> Any:
    """Parse potentially incomplete JSON from streaming tool call arguments.

    Always returns a valid object — never raises.
    """
    if not partial or not partial.strip():
        return {}
    try:
        return json.loads(partial)
    except json.JSONDecodeError:
        try:
            result = repair_json(partial, return_objects=True)
            return result if result else {}
        except Exception:
            return {}
