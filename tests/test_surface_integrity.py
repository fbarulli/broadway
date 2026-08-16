"""Human report surface integrity.

Asserts the tracked ``reports/`` surface is self-consistent:

* every relative markdown link resolves on disk,
* tracked ``.html``/``.png`` files stay under a size cap.

This test only reads; it never writes to ``reports/`` or ``artifacts/``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

HTML_CAP_BYTES = 5 * 1024 * 1024
PNG_CAP_BYTES = 2 * 1024 * 1024

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]*)\)")


def _tracked_files(prefix: str, suffix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.endswith(suffix)]


def _markdown_files() -> list[Path]:
    return [REPO_ROOT / rel for rel in _tracked_files("reports/", ".md")]


def test_report_markdown_links_resolve() -> None:
    broken: list[str] = []
    for md in _markdown_files():
        text = md.read_text(encoding="utf-8")
        for target in _LINK_RE.findall(text):
            target = target.strip()
            if not target:
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "broken markdown links in reports/:\n" + "\n".join(broken)


def test_report_html_under_size_cap() -> None:
    for rel in _tracked_files("reports/", ".html"):
        path = REPO_ROOT / rel
        size = path.stat().st_size
        assert size < HTML_CAP_BYTES, (
            f"{rel} is {size} bytes, exceeds {HTML_CAP_BYTES} byte cap"
        )


def test_report_png_under_size_cap() -> None:
    for rel in _tracked_files("reports/", ".png"):
        path = REPO_ROOT / rel
        size = path.stat().st_size
        assert size < PNG_CAP_BYTES, (
            f"{rel} is {size} bytes, exceeds {PNG_CAP_BYTES} byte cap"
        )
