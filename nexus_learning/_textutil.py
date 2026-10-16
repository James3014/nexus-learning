"""Shared text helpers.

Only helpers that are behaviorally identical across modules live here. Other
modules keep their own ``_text`` because their error type/message (which is
part of their contract) or their coercion semantics differ.
"""

from __future__ import annotations

from typing import Any


def require_non_empty_text(value: Any, field: str) -> str:
    """Return ``value.strip()``; raise ``ValueError("<field> must be a non-empty string")``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
