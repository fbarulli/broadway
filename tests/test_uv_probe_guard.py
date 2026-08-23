"""Guard against the 2026-08-23 uv editable-rebuild probe stall.

Plain ``uv run`` re-syncs whenever pyproject.toml changes; the re-sync rebuilds
the editable install, whose setuptools package discovery walks
``[tool.setuptools.packages.find] where = ["src", "."]`` with
``followlinks=True``. The gitignored ``deepseek-harness/`` checkout (DSH
harness) contains node_modules symlink loops, so that walk never terminates
(100% CPU for 30+ min). The fix: exclude the foreign tree from discovery.

This guard fails the moment the exclude is removed or weakened — the stall
would silently come back on the next pyproject.toml change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# setuptools prunes a discovered dir when fnmatch(f"{package}*", pattern)
# matches; "deepseek-harness*" covers the foreign tree itself and everything
# beneath it (subpackages are pruned the same way).
_EXCLUDE_REQUIRED = "deepseek-harness*"


def _find_exclude() -> list[str]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    return list(find.get("exclude", []))


def test_package_discovery_excludes_deepseek_harness() -> None:
    exclude = _find_exclude()
    assert _EXCLUDE_REQUIRED in exclude, (
        "[tool.setuptools.packages.find] must keep excluding the gitignored "
        "deepseek-harness/ checkout: its node_modules symlink loops make the "
        "uv editable-rebuild probe (setuptools discovery, followlinks=True) "
        f"spin at 100% CPU for 30+ min whenever pyproject.toml changes. "
        f"current exclude: {exclude}"
    )


def test_package_discovery_still_walks_repo_root() -> None:
    """The root-level ``project`` package keeps discovery bound to the repo
    root — which is exactly why the deepseek-harness exclusion is required."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert "." in find.get("where", [])
