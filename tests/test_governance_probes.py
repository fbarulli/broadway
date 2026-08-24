"""Governance probes (TIERREV-1) — mechanical checks over repo docs.

Four probe families, each shipped as a pure function plus two kinds of tests:

* LIVE probes run against the working tree and must stay green at HEAD.
* FALSIFIABILITY tests seed corruption into ``tmp_path`` copies of the real
  documents and prove the probe function goes RED — testing the PROBE against
  fixtures, never the live files, so live probes stay green by construction.

Probe vocabulary calibrated against the live corpus (see TIERREV-1 report):
backticked-path tokens are whitespace-free and resolve literally, by wildcard
glob, or by basename anywhere in the pruned tree; historical lines (containing
deleted/historic/b15f66e/once) are exempt; the declared agent-ID namespace
matches role vocabulary agent/adversar*/reviewer/synthesis within 80 chars.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tier_classifier import CHECKLIST, FULL, classify

ROOT = Path(__file__).resolve().parents[1]

BACKTICKED_PATH = re.compile(r"`([^\s`]+)\.(py|sh|md|ya?ml|toml|txt|json|env|cfg|ini)`")
HEX8 = re.compile(r"\b[0-9a-f]{8}\b")
SEP_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
HISTORICAL_MARKERS = ("deleted", "historic", "b15f66e", "once")
ROLE_VOCABULARY = re.compile(r"agent|adversar|reviewer|synthesis", re.IGNORECASE)
COV_FLAG = re.compile(r"--cov-fail-under\s*[=:]\s*(\d+)")
PERCENT = re.compile(r"(\d+)\s*%")


# --------------------------------------------------------------------------- #
# Probe (a): ledger well-formedness — markdown tables need a header row and
# consistent pipe counts. Separator-less pipe-ledger prose fragments (the
# reconstructed FIXES.md incident-log style) are not GFM tables and are
# deliberately out of scope.
# --------------------------------------------------------------------------- #
def probe_ledger_tables(text: str, source: str = "<text>") -> None:
    """Every markdown table needs a header row and consistent pipe counts."""
    lines = text.splitlines()
    block: list[tuple[int, str]] = []

    def flush(rows: list[tuple[int, str]]) -> None:
        sep_positions = [i for i, (_, ln) in enumerate(rows) if SEP_ROW.match(ln)]
        if not sep_positions:
            return
        sep = sep_positions[0]
        if sep == 0:
            raise AssertionError(f"{source}:{rows[0][0]}: separator row with no header row above")
        header_no, header = rows[sep - 1]
        want = header.count("|")
        got_sep = rows[sep][1].count("|")
        if got_sep != want:
            raise AssertionError(
                f"{source}:{rows[sep][0]}: separator declares {got_sep} pipes, header {header_no} declares {want}"
            )
        for idx, (no, ln) in enumerate(rows):
            if idx in (sep - 1, sep) or SEP_ROW.match(ln):
                continue
            got = ln.count("|")
            if got != want:
                raise AssertionError(f"{source}:{no}: table row has {got} pipes, header declares {want}")

    for no, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("|"):
            block.append((no, ln))
            continue
        flush(block)
        block = []
    flush(block)


# --------------------------------------------------------------------------- #
# Probe (b): backticked path tokens in the contract docs resolve in-tree,
# unless their line is explicitly historical.
# --------------------------------------------------------------------------- #
_WALK_PRUNE = {
    ".git", ".uv-cache", ".mplconfig", ".pytest_cache", "__pycache__",
    "deepseek-harness", "node_modules", ".venv", "mlruns", "data", "artifacts",
    "results", "reports",
}


@lru_cache(maxsize=8)
def _repo_files(root: str) -> frozenset[str]:
    found = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _WALK_PRUNE]
        found.extend(names)
    return frozenset(found)


def _resolves(token: str, root: Path) -> bool:
    if "*" in token or "?" in token:
        return any(root.glob(token))
    if (root / token).exists():
        return True
    return _basename(token) in _repo_files(str(root))


def _basename(token: str) -> str:
    """Basename of a forward-slash path token."""
    return token.rsplit("/", 1)[-1]


def probe_backticked_paths(text: str, root: Path, source: str = "<text>") -> None:
    """Every backticked path token resolves in-tree or sits on a historic line."""
    for no, line in enumerate(text.splitlines(), 1):
        lower = line.lower()
        if any(marker in lower for marker in HISTORICAL_MARKERS):
            continue
        for name, ext in BACKTICKED_PATH.findall(line):
            token = f"{name}.{ext}"
            if not _resolves(token, root):
                raise AssertionError(f"{source}:{no}: backticked path `{token}` does not resolve in-tree")


# --------------------------------------------------------------------------- #
# Probe (c): 8-hex tokens are resolvable revisions or declared agent IDs.
# --------------------------------------------------------------------------- #
@cache
def _git_resolves(token: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{token}^{{commit}}"],
        capture_output=True, cwd=ROOT, check=False,
    )
    return proc.returncode == 0


def probe_hex_tokens(
    text: str,
    resolver: Callable[[str], bool],
    *,
    window: int = 80,
    source: str = "<text>",
) -> None:
    """Every 8-hex token resolves as a revision or is a declared agent ID."""
    for match in HEX8.finditer(text):
        token = match.group(0)
        if resolver(token):
            continue
        lo, hi = max(0, match.start() - window), min(len(text), match.end() + window)
        if ROLE_VOCABULARY.search(text[lo:hi]):
            continue
        raise AssertionError(
            f"{source}: 8-hex token {token} is neither a resolvable revision nor "
            f"inside the declared agent-ID namespace (role vocabulary within {window} chars)"
        )


# --------------------------------------------------------------------------- #
# Probe (d): the coverage floor quoted in docs equals the gate-script floor.
# --------------------------------------------------------------------------- #
def probe_coverage_floor(docs: dict[str, str], ci_script_text: str) -> None:
    """Every quoted coverage-floor number equals scripts/run_local_ci.sh's."""
    flagged = COV_FLAG.search(ci_script_text)
    if not flagged:
        raise AssertionError("gate script declares no --cov-fail-under floor")
    floor = flagged.group(1)
    for name, text in docs.items():
        quoted: set[str] = set(COV_FLAG.findall(text))
        for line in text.splitlines():
            if "coverage" in line.lower():
                quoted.update(PERCENT.findall(line))
        wrong = sorted(n for n in quoted if n != floor)
        if wrong:
            raise AssertionError(
                f"{name}: quotes coverage floor(s) {wrong} but the gate script owns {floor}"
            )


# --------------------------------------------------------------------------- #
# LIVE probes — green at HEAD is the gate.
# --------------------------------------------------------------------------- #
LEDGERS = ["FIXES.md", "DECISIONS.md", "agents/ledger/STATE.md"]
CONTRACT_DOCS = ["MAIN_AGENT_CONTRACT.md", "WORKER_CONTRACT.md", "CONTRACT_TEMPLATE.md"]


def test_probe_a_live_ledger_tables_wellformed() -> None:
    for name in LEDGERS:
        probe_ledger_tables((ROOT / name).read_text(encoding="utf-8"), source=name)


def test_probe_b_live_contract_paths_resolve() -> None:
    for name in CONTRACT_DOCS:
        probe_backticked_paths((ROOT / name).read_text(encoding="utf-8"), ROOT, source=name)


def test_probe_c_live_8hex_tokens_declared_or_resolvable() -> None:
    for name in ["FIXES.md", "DECISIONS.md"]:
        probe_hex_tokens((ROOT / name).read_text(encoding="utf-8"), _git_resolves, source=name)


def test_probe_d_live_floor_quotes_match_gate_script() -> None:
    docs = {name: (ROOT / name).read_text(encoding="utf-8") for name in ["README.md", "SKLEARN_PIPELINES.md"]}
    probe_coverage_floor(docs, (ROOT / "scripts/run_local_ci.sh").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# FALSIFIABILITY — seeded corruption on tmp_path copies proves RED.
# --------------------------------------------------------------------------- #
def _seeded_copy(tmp_path: Path, name: str, mutate: Callable[[list[str]], list[str]]) -> str:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    (tmp_path / "seeded.md").write_text("\n".join(mutate(lines)) + "\n", encoding="utf-8")
    return (tmp_path / "seeded.md").read_text(encoding="utf-8")


def test_probe_a_red_body_row_loses_pipe(tmp_path: Path) -> None:
    def steal_pipe(lines: list[str]) -> list[str]:
        sep = next(i for i, ln in enumerate(lines) if SEP_ROW.match(ln))
        lines[sep + 1] = lines[sep + 1].rsplit("|", 1)[0]
        return lines

    seeded = _seeded_copy(tmp_path, "FIXES.md", steal_pipe)
    with pytest.raises(AssertionError, match="header declares"):
        probe_ledger_tables(seeded, source="seeded-FIXES")


def test_probe_a_red_separator_count_mismatch(tmp_path: Path) -> None:
    def shrink_separator(lines: list[str]) -> list[str]:
        sep = next(i for i, ln in enumerate(lines) if SEP_ROW.match(ln))
        lines[sep] = lines[sep].replace("|---|---|---|", "|---|---|", 1)
        return lines

    seeded = _seeded_copy(tmp_path, "agents/ledger/STATE.md", shrink_separator)
    with pytest.raises(AssertionError, match="separator declares"):
        probe_ledger_tables(seeded, source="seeded-STATE")


def test_probe_b_red_phantom_path_in_contract(tmp_path: Path) -> None:
    def rename_dataflow(lines: list[str]) -> list[str]:
        hit = next(i for i, ln in enumerate(lines) if "`dataflow.md`" in ln)
        lines[hit] = lines[hit].replace("`dataflow.md`", "`dataflow_phantom.md`")
        return lines

    seeded = _seeded_copy(tmp_path, "MAIN_AGENT_CONTRACT.md", rename_dataflow)
    with pytest.raises(AssertionError, match="does not resolve in-tree"):
        probe_backticked_paths(seeded, ROOT, source="seeded-MAC")


def test_probe_c_red_unattributed_8hex_token(tmp_path: Path) -> None:
    def append_mystery(lines: list[str]) -> list[str]:
        return lines + ["", "Mystery reference cafe1234 ends here.", ""]

    seeded = _seeded_copy(tmp_path, "DECISIONS.md", append_mystery)

    def everything_except_seed(token: str) -> bool:
        return token != "cafe1234"  # deterministic stub resolver, no git dependence

    with pytest.raises(AssertionError, match="declared agent-ID namespace"):
        probe_hex_tokens(seeded, everything_except_seed, source="seeded-DECISIONS")


def test_probe_d_red_stale_floor_quote(tmp_path: Path) -> None:
    script = (ROOT / "scripts/run_local_ci.sh").read_text(encoding="utf-8")

    def stale_floor(lines: list[str]) -> list[str]:
        return [ln.replace("cov-fail-under=95", "cov-fail-under=94") for ln in lines]

    seeded = _seeded_copy(tmp_path, "SKLEARN_PIPELINES.md", stale_floor)
    with pytest.raises(AssertionError, match="owns 95"):
        probe_coverage_floor({"SKLEARN_PIPELINES.md": seeded}, script)
    with pytest.raises(AssertionError, match="owns 95"):
        probe_coverage_floor({"README.md": "coverage gate holds 90% for new modules.\n"}, script)


# --------------------------------------------------------------------------- #
# Classifier pins (TIERREV-1 deliverable 1) — behavior locked at this level.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("line", "trigger"),
    [
        ("landed as 3db7b4be today", "sha-like-token"),
        ("coverage floor raised to 95 percent", "gate-number"),
        ("see `scripts/run_local_ci.sh` for gates", "backticked-path"),
        ("uv run pytest -q", "shell-ish-line"),
        ("```python", "fenced-code"),
    ],
)
def test_classifier_each_trigger_fires(line: str, trigger: str) -> None:
    result = classify(["README.md"], [line])
    assert result["tier"] == FULL
    assert trigger in result["reasons"][0]


def test_classifier_governance_file_forces_full() -> None:
    result = classify(["DECISIONS.md", "agents/ledger/STATE.md"], [])
    assert result["tier"] == FULL
    assert result["reasons"][0] == "executed-prose governance file — probes + scoped adversary mandatory"


def test_classifier_behavior_surface_full() -> None:
    for path in ["src/broadway/x.py", "tests/test_x.py", "docker-compose.yml", "Dockerfile",
                 "configs/experiment/taxi.yaml", "k8s/deploy.yaml", "tools/run.sh", "uv.lock"]:
        result = classify([path], [])
        assert result["tier"] == FULL, path
        assert "behavior surface" in result["reasons"]


def test_classifier_descriptive_prose_is_checklist() -> None:
    result = classify(["README.md"], ["plain words only"])
    assert result == {"tier": CHECKLIST, "reasons": ["descriptive prose, zero triggers"]}


def test_classifier_unknown_and_empty_default_up() -> None:
    assert classify(["demo/blob.bin"], [])["reasons"][0].startswith("unclassified path(s)")
    assert classify([], [])["tier"] == FULL


def test_classifier_mixed_signals_aggregate_to_full() -> None:
    result = classify(["DECISIONS.md", "src/x.py"], ["```"])
    assert result["tier"] == FULL
    joined = " ".join(result["reasons"])
    assert "governance files: DECISIONS.md" in joined and "behavior surface" in joined
    assert "fenced-code" in joined


def test_classifier_cli_end_to_end() -> None:
    payload = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1,2 @@\n context\n+new descriptive sentence\n"
    )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/tier_classifier.py")],
        input=payload, capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"tier": CHECKLIST, "reasons": ["descriptive prose, zero triggers"]}
