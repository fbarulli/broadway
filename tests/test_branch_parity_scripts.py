"""Branch-parity surface extension — scripts/ joins the SHARED surface.

Single era vocabulary (D16): the checker self-gates on the COMMITTED
``.github/parity-era.env`` (``PARITY_ERA=dev|main``), and this suite reads
the same file — there is exactly one era declaration. The old
``PARITY_MAIN_DAY`` os.environ dialect is deleted: CI sets no environment
variables, so environ-based gates re-create the hole they were meant to
close. Two guards:

* ``test_parity_surface_includes_scripts`` (always runs) — the checker's
  ``SHARED`` list and header comment must name ``scripts/``, so a future
  patch cannot silently drop the entry.
* ``test_scripts_diff_empty_vs_main`` (era-file gated) — runs its body ONLY
  when the committed era is ``main``; otherwise it skips. On ``dev`` the
  sklearn-vs-main shared surface diverges by design until the human-declared
  main-day flip (``PARITY_ERA=dev`` → ``main`` in ONE commit citing D16c).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_branch_parity.sh"
ERA_FILE = REPO_ROOT / ".github" / "parity-era.env"


def _shared_entries() -> list[str]:
    """Parse the checker's SHARED array body into its entry strings."""
    text = CHECKER.read_text(encoding="utf-8")
    body = text.split("SHARED=(", 1)[1].split(")", 1)[0]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _committed_era() -> str:
    """Parse PARITY_ERA from the committed era file in the working tree."""
    for raw_line in ERA_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("PARITY_ERA="):
            continue
        value = line.split("=", 1)[1].split("#", 1)[0].strip()
        if value:
            return value
    raise AssertionError(
        f"{ERA_FILE.relative_to(REPO_ROOT)} carries no parsable PARITY_ERA= "
        "line — it is the single era declaration (D16a)"
    )


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


def test_scripts_diff_empty_vs_main() -> None:
    """The sklearn-vs-main scripts/ diff must be empty (era-gated, main day).

    Mirrors the checker's own comparison (``git diff --exit-code --quiet
    origin/main origin/taxi -- scripts/``); sklearn tracks taxi, so this
    asserts the pair origin/main vs origin/taxi is in sync for ``scripts/``.

    Gating reads the COMMITTED ``.github/parity-era.env`` from the repo
    working tree — never an environment variable. The body runs only when
    the parsed era is ``main`` (declared by the human on main-day); on any
    other value the test skips with the observed era in the reason, keeping
    pre-main-day CI green while making the gate auditable in test output.
    """
    era = _committed_era()
    if era != "main":
        pytest.skip(f"pre-main-day: era={era} (from .github/parity-era.env)")
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
