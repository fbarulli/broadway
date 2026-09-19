"""Canonical repository-root resolution.

The ONLY sanctioned way to locate repo-relative paths: every caller uses
repo_root() instead of inlining ``Path(__file__).resolve().parents[...]``.
File-anchored (no config, no cwd dependence): this module lives at
``src/broadway/paths.py``, so the root is always two levels up. Reproducible
on every checkout, worktree, and branch — taxi and main share it via parity.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Return the canonical repository root."""
    return _REPO_ROOT
