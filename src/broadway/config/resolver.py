"""Resolve ${paths}, interpolate env vars, expand globs."""

from __future__ import annotations

import glob
import os
from typing import Any


def _resolve_string(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _resolve_globs(value: str) -> list[str]:
    return sorted(glob.glob(value, recursive=True))


def resolve_values(obj: Any) -> Any:
    """Recursively walk dicts/lists and resolve ${ENV} placeholders in strings."""
    if isinstance(obj, dict):
        return {key: resolve_values(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [resolve_values(item) for item in obj]
    if isinstance(obj, str):
        return _resolve_string(obj)
    return obj
