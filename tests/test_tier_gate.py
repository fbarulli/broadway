"""Tier-gate library law — the D34/D35 grammar + the 2026-09-03 ledger batch law.

scripts/tier_gate.sh is the ONE law library shared by scripts/ship.sh and the
pre-push hook (one law, two doors). Until 2026-09-03 it had ZERO automated
validation (acknowledged debt in its registry row); this suite pins:

* tg_check_message — the Tier trailer vocabulary + the FULL-tier Reviewer
  resolution rule (row id OR authority registration, echoed in the message).
* tg_batch_adds_row / tg_batch_terminal / tg_ledger_batch — the ledger batch
  law over REAL repo history: a batch must add >=1 STATE row AND terminally
  disposition >=1. History is the fixture: commits that only added rows and
  commits that only closed rows exist in the log, and the helper must
  classify each correctly (RED on add-only, RED on close-only, and the law
  is PASS only when a batch does both).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TG = REPO / "scripts" / "tier_gate.sh"


def _tg(*funcs: str, stdin: str | None = None, msg: str | None = None) -> tuple[int, str]:
    """Source the library, run funcs (joined by newlines), return (rc, out)."""
    body = f". {TG}\n" + "\n".join(funcs)
    env_prefix = f'TG_TEST_MSG=$(cat <<\'EOF\'\n{msg}\nEOF\n)\n' if msg is not None else ""
    result = subprocess.run(
        ["bash", "-c", env_prefix + body],
        cwd=REPO, capture_output=True, text=True, timeout=60, check=False,
        input=stdin,
    )
    return result.returncode, result.stdout + result.stderr


def _run(msg: str) -> tuple[int, str]:
    return _tg(
        'reason="$(tg_check_message "$(_tg_events_at HEAD)" "$TG_TEST_MSG")"; '
        'echo "$reason"; [ -n "$reason" ] && exit 1 || exit 0',
        msg=msg,
    )


# --- Tier trailer grammar ----------------------------------------------------

def test_missing_tier_trailer_refused() -> None:
    rc, out = _run("subject\n\nbody\n")
    assert rc != 0 and "missing required 'Tier:' trailer" in out, out


def test_bad_tier_word_refused() -> None:
    rc, out = _run("subject\n\nTier: full\n")
    assert rc != 0 and "bad tier word 'full'" in out, out  # case-sensitive


def test_fast_tier_accepted() -> None:
    rc, out = _run("subject\n\nTier: FAST\n")
    assert rc == 0, out


def test_full_tier_requires_resolvable_reviewer() -> None:
    rc, out = _run("subject\n\nTier: FULL\nReviewer: none\n")
    assert rc != 0 and "Reviewer" in out, out


def test_full_tier_accepts_registered_authority() -> None:
    # 39de4245 is the hash-pinned reviewer-authority registration row in EVENTS.
    rc, out = _run("subject\n\nTier: FULL\nReviewer: 39de4245\n")
    assert rc == 0, out


# --- Ledger batch law over real history ---------------------------------------

def _batch(base: str, sha: str) -> tuple[int, str]:
    return _tg(f"tg_ledger_batch {base} {sha} && echo BATCH-PASS || echo BATCH-REFUSED")


def test_add_only_batch_refused_by_law() -> None:
    """A batch that only ADDS rows (never closes) must be refused.

    987a0b0 added 3 rows and closed none — the exact 'nothing ever closes'
    pathology the law exists to end.
    """
    _rc, out = _batch("987a0b0~1", "987a0b0")
    assert "BATCH-REFUSED" in out and "dispositions no STATE row" in out, out


def test_close_only_batch_refused_by_law() -> None:
    """A batch that only CLOSES a row (adds none) must be refused too."""
    _rc, out = _batch("7e48eca~1", "7e48eca")
    assert "BATCH-REFUSED" in out and "adds no STATE row" in out, out


def test_empty_batch_refused() -> None:
    """Zero-range batch: adds nothing, closes nothing — refused."""
    _rc, out = _batch("HEAD", "HEAD")
    assert "BATCH-REFUSED" in out, out


def test_law_passes_when_batch_does_both() -> None:
    """PASS requires BOTH acts in one batch — synthesized from real history.

    Two adjacent real commits, one adding a row and one closing a row, are
    merged into one batch range: the law must pass it (proving the check is
    genuinely two-key, not a find-any match).
    """
    _rc, out = _batch("7e48eca~1", "987a0b0")  # closes -033, then adds 3 rows
    assert "BATCH-PASS" in out, out
