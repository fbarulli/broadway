"""Branch-parity surface extension — scripts/ joins the SHARED surface.

Locks in CONTRACT FIX_3: the branch-parity checker must treat ``scripts/``
as part of the shared surface that must be byte-identical on origin/main and
origin/taxi, exactly like ``src/`` and ``tests/``. Two guards:

* ``test_parity_surface_includes_scripts`` (always runs) — the checker's
  ``SHARED`` list and header comment must name ``scripts/``, so a future
  patch cannot silently drop the entry.
* ``test_scripts_diff_empty_vs_main`` (main-day-gated) — the sklearn-vs-main
  ``scripts/`` diff must be empty. Pre-main-day this is false by design
  (origin/main intentionally lacks two scripts), so the test is skipped
  unless the human sets ``PARITY_MAIN_DAY=1`` on main-day.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_branch_parity.sh"


def _shared_entries() -> list[str]:
    """Parse the checker's SHARED array body into its entry strings."""
    text = CHECKER.read_text(encoding="utf-8")
    body = text.split("SHARED=(", 1)[1].split(")", 1)[0]
    return [line.strip() for line in body.splitlines() if line.strip()]


def test_parity_surface_includes_scripts() -> None:
    """The checker's watched surface names scripts/ (directory + header)."""
    entries = _shared_entries()
    assert "scripts/" in entries, (
        f"{CHECKER.relative_to(REPO_ROOT)} SHARED list lacks a 'scripts/' entry — "
        "scripts/ must join the parity surface (CONTRACT FIX_3)"
    )
    # The header comment must name scripts/ among the shared surfaces.
    header = CHECKER.read_text(encoding="utf-8").split("SHARED=(", 1)[0]
    assert "scripts/" in header, (
        f"{CHECKER.relative_to(REPO_ROOT)} header comment does not name scripts/ "
        "as part of the shared surface"
    )


@pytest.mark.skipif(
    os.environ.get("PARITY_MAIN_DAY") != "1",
    reason=(
        "pre-main-day: origin/main intentionally lacks scripts/check_e2e_determinism.sh "
        "and scripts/check_champion_manifest.sh; runs only on main-day (PARITY_MAIN_DAY=1)"
    ),
)
def test_scripts_diff_empty_vs_main() -> None:
    """The sklearn-vs-main scripts/ diff must be empty (main-day gate).

    Mirrors the checker's own comparison (``git diff --exit-code --quiet
    origin/main origin/taxi -- scripts/``); sklearn tracks taxi, so this
    asserts the pair origin/main vs origin/taxi is in sync for ``scripts/``.

    The ``PARITY_MAIN_DAY`` marker is set ONLY by the human on main-day — the
    worker and CI must never set it. Pre-main-day origin/main intentionally
    lacks ``scripts/check_e2e_determinism.sh`` and
    ``scripts/check_champion_manifest.sh``, so this assertion is false by
    design and the test skips (not fails), keeping pre-main-day CI green.
    """
    result = subprocess.run(
        ["git", "diff", "--exit-code", "--quiet", "origin/main", "origin/taxi", "--", "scripts/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        "scripts/ differs between origin/main and origin/taxi — origin/main is "
        "missing scripts/check_e2e_determinism.sh and/or "
        "scripts/check_champion_manifest.sh (sklearn tracks taxi); run the "
        "human-gated main-day sync so the shared surface is identical"
        + (f"\ngit stderr: {result.stderr.strip()}" if result.stderr.strip() else "")
    )
