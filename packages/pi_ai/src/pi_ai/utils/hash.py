"""Fast deterministic hashing for internal ID normalization."""

import hashlib


def short_hash(value: str) -> str:
    """Generate a short deterministic hash (12-char hex)."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]
