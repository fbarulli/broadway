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
  taxi-vs-main shared surface diverges by design until the human-declared
  main-day flip (``PARITY_ERA=dev`` → ``main`` edited in ONE commit citing
  D16c/D21).
* ``test_f1b_guard_rejects_legacy_checker_without_era_marker`` — proves the
  F1b guard in ``scripts/run_local_ci.sh`` refuses to gate CI with a stale
  pre-D16/D21 checker (see its docstring for the mechanism).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from broadway.paths import repo_root

REPO_ROOT = repo_root()
CHECKER = REPO_ROOT / "scripts" / "check_branch_parity.sh"
RUN_CI = REPO_ROOT / "scripts" / "run_local_ci.sh"


def _whitelist_path() -> Path:
    """Resolve the shared-surface whitelist SSOT (configs/tooling.yaml -> file)."""
    import yaml

    tooling = REPO_ROOT / "configs" / "tooling.yaml"
    data = yaml.safe_load(tooling.read_text(encoding="utf-8"))
    return REPO_ROOT / data["shared_surface"]["file"]


def _shared_entries() -> list[str]:
    """Read the shared surface from the whitelist SSOT (no inline list)."""
    entries: list[str] = []
    for raw_line in _whitelist_path().read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


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


def test_parity_surface_includes_scripts() -> None:
    """The shared surface (whitelist SSOT) names scripts/; checker reads it."""
    entries = _shared_entries()
    assert "scripts/" in entries, (
        f"shared-surface whitelist {_whitelist_path().relative_to(REPO_ROOT)} lacks a "
        "'scripts/' entry — scripts/ must join the parity surface (CONTRACT FIX_3)"
    )
    # The checker must read the whitelist SSOT, not maintain an inline list.
    text = CHECKER.read_text(encoding="utf-8")
    assert "main_whitelist.txt" in text, (
        f"{CHECKER.relative_to(REPO_ROOT)} no longer reads the whitelist SSOT "
        "(deterministic-ops law: one list, read from file)"
    )
    assert "configs/tooling.yaml" in text, (
        f"{CHECKER.relative_to(REPO_ROOT)} must resolve the whitelist via "
        "configs/tooling.yaml shared_surface.file (no hardcoded values)"
    )
    # No inline SHARED=( hard list may remain (empty init + file load only).
    assert "SHARED=(\n" not in text and "SHARED=(" not in text.replace("SHARED=()", ""), (
        f"{CHECKER.relative_to(REPO_ROOT)} carries an inline SHARED list — "
        "the whitelist file is the single source"
    )
    # D21: the relocated env file must NOT linger as a maintained array line.
    assert ".github/parity-era.env" not in entries, (
        "SHARED still lists .github/parity-era.env after D21 inlined the era "
        "declaration into the checker — drop the line (zero lines to maintain)"
    )


def test_scripts_diff_empty_vs_main() -> None:
    """The taxi-vs-main scripts/ diff must be empty (era-gated, main day).

    Mirrors the checker's own comparison (``git diff --exit-code --quiet
    origin/main origin/taxi -- scripts/``); the track line is taxi, so this
    asserts the pair origin/main vs origin/taxi is in sync for ``scripts/``.

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
        "scripts/check_champion_manifest.sh (track line is taxi); run the "
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
        'if [[ "$1 $2" == "show refs/remotes/origin/taxi:'
        'scripts/check_branch_parity.sh" ]]; then\n'
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
    on PATH, so ``git show refs/remotes/origin/taxi:…`` yields our
    fixture instead of the network truth. Choice documented per contract:
    extract-and-execute over replicate, so the test cannot outlive the
    guard's actual semantics.

    * legacy fixture (pre-D16 shape: no ``^PARITY_ERA=`` line) ⇒ guard exits
      non-zero naming the legacy checker — RED demonstrated.
    * post-D21 control fixture (marker present) ⇒ guard proceeds past the
      marker check and the minimal declarations execute cleanly — proving
      the verdict above is caused by the missing marker alone.
    """
    fixture = tmp_path / "checker_under_test.sh"
    marker_block = (
        "PARITY_ERA=dev\n"
        "PARITY_TRACK_BRANCH=taxi\n"
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

    bin_dir = tmp_path / "bin"
    _git_show_stub(bin_dir, fixture)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run(
        ["bash", "-c", f"{_gate_parity_source()}\ngate_parity\n"],
        cwd=tmp_path,
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


def _gate_functions_source() -> str:
    """Extract gate_parity + gate_parity_anchor_bump from run_local_ci.sh.

    Same extract-and-execute rationale as _gate_parity_source: the test
    cannot outlive the guard's actual semantics.
    """
    text = RUN_CI.read_text(encoding="utf-8")
    assert "run parity gate_parity" in text, "parity gate unwired"
    chunks = []
    for name in ("gate_parity() {", "gate_parity_anchor_bump() {"):
        start = text.index(name)
        end = text.index("\n}", start)
        chunks.append(text[start : end + 2])
    return "\n".join(chunks)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
        timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout.strip()


def _scratch_repo_with_checker(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    """Scratch repo: origin/taxi pins an old checker, origin/main at new tip.

    Returns (repo, c0, c1) where c0 is the stale pinned anchor and c1 the
    origin/main tip. The pinned custody fails by construction (whitelist
    absent + stale anchor), isolating the fast-path verdict.
    """
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "base")
    c0 = _git(repo, "rev-parse", "HEAD")
    checker = (
        "# probe checker\n"
        "PARITY_ERA=dev\n"
        "PARITY_TRACK_BRANCH=taxi\n"
        "PARITY_ALLOWLIST=()\n"
        f"PARITY_MAIN_ANCHOR={c0}  # anchor\n"
        "# stale custody below: the pinned run always fails, isolating the\n"
        "# fast-path verdict (F1b covers live-checker semantics separately).\n"
        "echo 'ROGUE MAIN WRITE (stale probe anchor)' >&2\n"
        "exit 1\n"
    )
    (repo / "scripts").mkdir()
    (repo / "scripts" / "check_branch_parity.sh").write_text(checker, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "-m", "checker")
    c1 = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/taxi", c1)
    _git(repo, "update-ref", "refs/remotes/origin/main", c1)
    _git(repo, "update-ref", "refs/heads/taxi", c1)
    _git(repo, "checkout", "-q", "taxi")
    return repo, c0, c1


def test_anchor_bump_fast_path_admits_verifiable_reanchor(tmp_path: Path) -> None:
    """Anchor bumps must not need --no-verify when verifiably correct.

    Worktree checker differs from the pinned one ONLY on the anchor line
    and the new pin is exactly origin/main's tip: gate_parity must admit
    it — post-push custody holds by construction.
    """
    repo, c0, c1 = _scratch_repo_with_checker(tmp_path, "repo-ok")
    (repo / "scripts" / "check_branch_parity.sh").write_text(
        (repo / "scripts" / "check_branch_parity.sh").read_text(encoding="utf-8").replace(c0, c1),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-c", f"{_gate_functions_source()}\ngate_parity\n"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"fast path refused a verifiable bump\n{combined}"
    assert "anchor-bump fast path" in combined


def test_anchor_bump_fast_path_rejects_wrong_pin(tmp_path: Path) -> None:
    """A bump pointing anywhere but origin/main's tip stays refused."""
    repo, c0, c1 = _scratch_repo_with_checker(tmp_path, "repo-bad")
    assert c0 != c1
    result = subprocess.run(
        ["bash", "-c", f"{_gate_functions_source()}\ngate_parity\n"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0, "fast path admitted a bump missing origin/main"


def _custody_source() -> str:
    """Extract the LIVE custody() body from scripts/check_branch_parity.sh.

    Same extract-and-execute rationale as _gate_parity_source: the tests
    execute the checker's real custody logic (including the MAIN-SYNC
    transition rule) under bash in a tmp scratch repo, so they cannot
    outlive the checker's actual semantics. Fails loudly if custody()
    disappears or loses the transition marker.
    """
    text = CHECKER.read_text(encoding="utf-8")
    assert "custody" in text, "checker no longer defines custody()"
    start = text.index("custody() {")
    end = text.index("\n}", start)
    source = text[start : end + 2]
    assert "MAIN-SYNC TRANSITION" in source, (
        "live custody() carries no MAIN-SYNC transition rule — the checker "
        "must pass a tip one ratified sync ahead (parent==anchor + MAIN-SYNC "
        "subject) instead of ROGUE MAIN WRITE"
    )
    return source


def _custody_scratch(tmp_path: Path, name: str) -> tuple[Path, str]:
    """Scratch repo with one shared file; origin/main+taxi pinned at anchor.

    Returns (repo, anchor) where anchor is the commit both remotes point
    at. Callers advance origin/main to build transition / negative cases.
    No network: all refs are local update-ref pins.
    """
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "shared.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "-m", "base")
    anchor = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/taxi", anchor)
    _git(repo, "update-ref", "refs/remotes/origin/main", anchor)
    return repo, anchor


def _commit_shared(repo: Path, content: str, message: str) -> str:
    """Rewrite scripts/shared.txt, commit, move origin/main to the new tip."""
    (repo / "scripts" / "shared.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "-m", message)
    tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", tip)
    return tip


def _run_custody(repo: Path, anchor: str) -> subprocess.CompletedProcess[str]:
    """Execute the extracted custody() with a pinned anchor in the scratch repo."""
    script = (
        "set -euo pipefail\n"
        f"{_custody_source()}\n"
        'SHARED=("scripts/shared.txt")\n'
        f"PARITY_MAIN_ANCHOR={anchor}\n"
        "PARITY_TRACK_BRANCH=taxi\n"
        "PARITY_ALLOWLIST=()\n"
        "custody\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_custody_main_sync_transition_passes(tmp_path: Path) -> None:
    """TRANSITION: parent==anchor + MAIN-SYNC subject passes loudly."""
    repo, anchor = _custody_scratch(tmp_path, "repo-transition")
    tip = _commit_shared(repo, "v2\n", "MAIN-SYNC: ratified main-day sync")
    parent = _git(repo, "rev-parse", f"{tip}^1")
    assert parent == anchor
    result = _run_custody(repo, anchor)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        "custody rejected a provably one-ahead ratified sync "
        f"(parent==anchor, MAIN-SYNC subject)\noutput:\n{combined}"
    )
    assert "MAIN-SYNC TRANSITION" in combined, (
        f"transition pass lacks the loud notice line:\n{combined}"
    )


def test_custody_non_marker_still_fails(tmp_path: Path) -> None:
    """NON-TRANSITION: same one-ahead tip without the marker still fails."""
    repo, anchor = _custody_scratch(tmp_path, "repo-nonmarker")
    tip = _commit_shared(repo, "v2\n", "HOTFIX: unratified main write")
    parent = _git(repo, "rev-parse", f"{tip}^1")
    assert parent == anchor
    result = _run_custody(repo, anchor)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        "custody admitted a one-ahead main write with a non-MAIN-SYNC "
        f"subject\noutput:\n{combined}"
    )
    assert "ROGUE MAIN WRITE" in combined, (
        f"wrong failure mode (expected ROGUE MAIN WRITE):\n{combined}"
    )
    assert "MAIN-SYNC TRANSITION" not in combined


def test_custody_two_ahead_still_fails(tmp_path: Path) -> None:
    """NON-TRANSITION: tip two commits ahead fails even with MAIN-SYNC marks."""
    repo, anchor = _custody_scratch(tmp_path, "repo-twoahead")
    _commit_shared(repo, "v2\n", "MAIN-SYNC: ratified main-day sync")
    tip2 = _commit_shared(repo, "v3\n", "MAIN-SYNC: second sync")
    parent2 = _git(repo, "rev-parse", f"{tip2}^1")
    assert parent2 != anchor, "fixture error: tip is not two ahead of anchor"
    result = _run_custody(repo, anchor)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        "custody admitted a two-ahead tip (deeper history must fail exactly "
        f"as before)\noutput:\n{combined}"
    )
    assert "ROGUE MAIN WRITE" in combined, (
        f"wrong failure mode (expected ROGUE MAIN WRITE):\n{combined}"
    )
    assert "MAIN-SYNC TRANSITION" not in combined
