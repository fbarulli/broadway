"""Branch-parity surface extension — scripts/ joins the SHARED surface.

Single era vocabulary (D16): the checker self-gates on its OWN inline era
declaration — D21 relocated the former separate env file verbatim into
``scripts/check_branch_parity.sh`` ("zero array lines / no separate file"),
and this suite parses those same inline declarations from the checker text,
so there is exactly one era declaration and no second file to drift. The old
``PARITY_MAIN_DAY`` os.environ dialect is deleted: CI sets no environment
variables, so environ-based gates re-create the hole they were meant to
close. Three guards here:

* ``test_parity_surface_includes_scripts`` (always runs) — the checker's
  ``SHARED`` list and header comment must name ``scripts/``, so a future
  patch cannot silently drop the entry.
* ``test_scripts_diff_empty_vs_main`` (era-gated) — runs its body ONLY when
  the declared era is ``main``; otherwise it skips. On ``dev`` the
  track-vs-main shared surface diverges by design until the human-declared
  main-day flip (``PARITY_ERA=dev`` → ``main`` edited in ONE commit citing
  D16c/D21). The track branch is SAID ONCE — parsed from the checker's
  ``^PARITY_TRACK_BRANCH=`` declaration, never hard-coded here.
* ``test_f1b_guard_rejects_legacy_checker_without_era_marker`` — proves the
  F1b guard in ``scripts/run_local_ci.sh`` refuses to gate CI with a stale
  pre-D16/D21 checker (see its docstring for the mechanism).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_branch_parity.sh"
RUN_CI = REPO_ROOT / "scripts" / "run_local_ci.sh"

def _shared_entries() -> list[str]:
    """Parse the checker's SHARED array body into its entry strings."""
    text = CHECKER.read_text(encoding="utf-8")
    body = text.split("SHARED=(", 1)[1].split(")", 1)[0]
    return [line.strip() for line in body.splitlines() if line.strip()]

def _declared_era() -> str:
    """Parse PARITY_ERA from the checker's INLINE declaration (D21).

    Reads the ``^PARITY_ERA=`` line straight out of the checker text — the
    same marker the F1b staleness grep keys on — never an environment
    variable and never a second file.
    """
    for raw_line in CHECKER.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("PARITY_ERA="):
            continue
        value = raw_line.split("=", 1)[1].split("#", 1)[0].strip()
        if value:
            return value
        break
    raise AssertionError(
        f"{CHECKER.relative_to(REPO_ROOT)} carries no parsable inline "
        "`^PARITY_ERA=` declaration — it is the single era declaration "
        "(D16a/D21)"
    )

def _declared_track_branch() -> str:
    """Parse PARITY_TRACK_BRANCH from the checker's INLINE declaration.

    Say-it-once companion to _declared_era: the same single declaration every
    runtime consumer (F1b pin, main-day sync, this suite) reads — never a
    second copy in this file.
    """
    for raw_line in CHECKER.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("PARITY_TRACK_BRANCH="):
            continue
        value = raw_line.split("=", 1)[1].split("#", 1)[0].strip()
        if value:
            return value
        break
    raise AssertionError(
        f"{CHECKER.relative_to(REPO_ROOT)} carries no parsable inline "
        "`^PARITY_TRACK_BRANCH=` declaration — it is the single track-name "
        "declaration (say-it-once law 2026-09-02)"
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
    # D21: the relocated env file must NOT linger as a maintained array line.
    assert ".github/parity-era.env" not in entries, (
        "SHARED still lists .github/parity-era.env after D21 inlined the era "
        "declaration into the checker — drop the line (zero lines to maintain)"
    )

def test_scripts_diff_empty_vs_main() -> None:
    """The euromonitor-vs-main scripts/ diff must be empty (era-gated, main day).

    Mirrors the checker's own comparison (``git diff --exit-code --quiet
    origin/main origin/$PARITY_TRACK_BRANCH -- scripts/``): the track name is
    read from the single ``^PARITY_TRACK_BRANCH=`` declaration (say-it-once
    law 2026-09-02), so this test survives rename-day without edits.

    Gating reads the INLINE ``^PARITY_ERA=`` declaration in the checker
    working-tree text (D21) — never an environment variable. The body runs
    only when the parsed era is ``main`` (declared by the human on main-day);
    on any other value the test skips with the observed era in the reason,
    keeping pre-main-day CI green while making the gate auditable in output.
    """
    era = _declared_era()
    if era != "main":
        pytest.skip(
            f"pre-main-day: era={era} (inline declaration in "
            "scripts/check_branch_parity.sh)"
        )
    track = _declared_track_branch()
    result = subprocess.run(
        ["git", "diff", "--exit-code", "--quiet", "origin/main", f"origin/{track}", "--", "scripts/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        "scripts/ differs between origin/main and the track ref — origin/main is "
        "missing scripts/check_e2e_determinism.sh and/or "
        "scripts/check_champion_manifest.sh; run the "
        "human-gated main-day sync so the shared surface is identical"
        + (f"\ngit stderr: {result.stderr.strip()}" if result.stderr.strip() else "")
    )

def _gate_parity_source() -> str:
    """Extract the REAL gate_parity function body from run_local_ci.sh.

    Design choice (per contract option A): rather than replicating the
    guard's grep assertion in Python — which would drift silently if the
    guard changes — this pulls the live function text out of the script at
    test time and executes it under bash. The extraction fails loudly if
    the function or its call site disappears.
    """
    text = RUN_CI.read_text(encoding="utf-8")
    assert "run parity gate_parity" in text, (
        "run_local_ci.sh no longer routes the parity gate through gate_parity "
        "(F1b guard unwired)"
    )
    start = text.index("gate_parity() {")
    end = text.index("\n}", start)
    return text[start : end + 2]

def _git_show_stub(bin_dir: Path, fixture: Path) -> None:
    """Install a PATH-shim `git` whose `show <track-ref>:<checker>` emits fixture."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "git"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "show" && "$2" == refs/remotes/origin/trackstub:'
        'scripts/check_branch_parity.sh ]]; then\n'
        f'  cat "{fixture}"\n'
        "  exit 0\n"
        "fi\n"
        'echo "unexpected git invocation: $*" >&2\n'
        "exit 128\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)

@pytest.mark.parametrize("with_marker", [False, True], ids=["legacy", "post-D21"])
def test_f1b_guard_rejects_legacy_checker_without_era_marker(
    tmp_path: Path, with_marker: bool
) -> None:
    """NEGATIVE (F1b): the guard must refuse a checker lacking ``^PARITY_ERA=``.

    Mechanism: the real ``gate_parity`` body extracted from
    ``scripts/run_local_ci.sh`` is executed under bash with a stub ``git``
    on PATH, so ``git show refs/remotes/origin/<track>:…`` yields our
    fixture instead of the network truth. The track name is parsed by the
    guard itself from the tree-local checker at REPO_ROOT (say-it-once
    law), so the harness copies the real checker text to the sandbox cwd.
    Choice documented per contract: extract-and-execute over replicate, so
    the test cannot outlive the guard's actual semantics.

    * legacy fixture (pre-D16 shape: no ``^PARITY_ERA=`` line) ⇒ guard exits
      non-zero naming the legacy checker — RED demonstrated.
    * post-D21 control fixture (marker present) ⇒ guard proceeds past the
      marker check and the minimal declarations execute cleanly — proving
      the verdict above is caused by the missing marker alone.
    """
    fixture = tmp_path / "checker_under_test.sh"
    marker_block = (
        "PARITY_ERA=dev\n"
        f"PARITY_TRACK_BRANCH={_declared_track_branch()}\n"
        "PARITY_ALLOWLIST=()\n"
        f"PARITY_MAIN_ANCHOR={'a' * 40}\n"
    ) if with_marker else ""
    fixture.write_text(
        "#!/usr/bin/env bash\n"
        "# simulated track-ref checker content\n"
        "set -euo pipefail\n"
        + marker_block,
        encoding="utf-8",
    )
    # Say-it-once: the guard parses the track name from a tree-local checker
    # named scripts/check_branch_parity.sh relative to ITS cwd — plant a
    # sandbox copy of the REAL checker whose track declaration is rewritten
    # to the stub ref name, so parse→stub is a closed loop (and the test
    # stays rename-proof: no real branch name appears anywhere here).
    sandbox = tmp_path / "repo"
    (sandbox / "scripts").mkdir(parents=True, exist_ok=True)
    sandbox_checker = CHECKER.read_text(encoding="utf-8").replace(
        f"PARITY_TRACK_BRANCH={_declared_track_branch()}",
        "PARITY_TRACK_BRANCH=trackstub",
    )
    assert "PARITY_TRACK_BRANCH=trackstub" in sandbox_checker  # rewrite landed
    (sandbox / "scripts" / "check_branch_parity.sh").write_text(
        sandbox_checker, encoding="utf-8"
    )

    bin_dir = tmp_path / "bin"
    _git_show_stub(bin_dir, fixture)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        ["bash", "-c", f"{_gate_parity_source()}\ngate_parity\n"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    combined = result.stdout + result.stderr

    if with_marker:
        assert result.returncode == 0, (
            "control failed: guard rejected a post-D21 checker carrying the "
            f"^PARITY_ERA= marker\noutput:\n{combined}"
        )
    else:
        assert result.returncode != 0, (
            f"F1b HOLE: gate_parity accepted a pre-D16 legacy checker with no "
            f"^PARITY_ERA= marker\noutput:\n{combined}"
        )
        assert "legacy pre-D16" in combined, (
            f"guard failed for the wrong reason:\n{combined}"
        )
