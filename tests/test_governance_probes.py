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
USER-MVP pilot event-ids are a THIRD namespace (F4′, senior 3813c37c): a
FULL-LINE ``EVENT: issues/<n>#issuecomment-<m> event-id <8hex>`` line's token
is valid only via a UNIQUE row in STATE.md's normative ``## EVENTS`` table —
role vocabulary grants NO escape inside that grammar. Probe C scans FIXES,
DECISIONS and STATE; inside STATE.md only shape-matched registry rows between
the ``## EVENTS`` heading and the next ``##`` heading are exempt (section-
scoped); every other hex-bearing line stays in scan.
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
HEX8 = re.compile(r"\b[0-9a-f]{8}\b")  # known lookalike, OUT OF SCOPE: reports/audit/* carries decimal join-counts with accidental hex shape (e.g. 17091666) — reports/ is outside probe-C scan scope by design
# Canonical EVENT-line grammar (senior 3813c37c, full-line anchored,
# no-trailing-newline form): <hex8> expands to the named group (?P<eid>...)
# so span extraction shares these exact bytes. Kept on ONE source line so
# the byte-contiguity pin below is meaningful.
EVENT_GRAMMAR_CANONICAL = r"^EVENT:\s*issues/(?P<issue>\d+)#issuecomment-(?P<comment>\d+)\s+event-id\s*(?P<eid>[0-9a-f]{8})\s*$"
EVENT_LINE = re.compile(EVENT_GRAMMAR_CANONICAL, re.MULTILINE)
# A line belongs to the event-id namespace ONLY when nothing but whitespace
# follows the token.
EVENTS_HEADER = ("event-id", "issue", "comment-id", "created_at", "type", "supersedes")
REGISTRY_ROW_SHAPE = re.compile(
    r"^\s*\|\s*(?P<eid>[0-9a-f]{8})\s*\|\s*issues/(?P<issue>\d+)#issuecomment-(?P<cid>\d+)\s*\|"
    r"\s*(?P=cid)\s*\|\s*\d{4}-\d{2}-\d{2}T[\d:.+Z-]+\s*\|\s*[\w-]+\s*\|\s*(?:-|[0-9a-f]{8})\s*\|\s*$"
)
SEP_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
HISTORICAL_MARKERS = ("deleted", "historic", "b15f66e", "once")
ROLE_VOCABULARY = re.compile(r"agent|adversar|reviewer|synthesis|senior", re.IGNORECASE)
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


def _parse_event_registry(state_text: str) -> frozenset[str]:
    """8-hex event-ids holding a resolution row under STATE.md's ## EVENTS."""
    ids: set[str] = set()
    in_events = False
    for line in state_text.splitlines():
        if line.startswith("#"):
            in_events = line.strip() == "## EVENTS"
            continue
        if not in_events or not line.lstrip().startswith("|"):
            continue
        first_cell = line.strip().strip("|").split("|")[0].strip()
        if HEX8.fullmatch(first_cell):
            ids.add(first_cell)
    return frozenset(ids)


@cache
def _registered_event_ids() -> frozenset[str]:
    return _parse_event_registry((ROOT / "agents/ledger/STATE.md").read_text(encoding="utf-8"))


def probe_event_registry_schema(state_text: str, source: str = "<state>") -> None:
    """Normative ## EVENTS table: exact header, hex8 ids, UNIQUE event-ids."""
    seen: dict[str, int] = {}
    in_events = False
    header_seen = False
    for no, line in enumerate(state_text.splitlines(), 1):
        if line.startswith("#"):
            in_events = line.strip() == "## EVENTS"
            continue
        if not in_events or not line.lstrip().startswith("|") or SEP_ROW.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header_seen:
            if tuple(cells) != EVENTS_HEADER:
                raise AssertionError(
                    f"{source}:{no}: ## EVENTS header declares {cells} but the normative "
                    f"schema is {list(EVENTS_HEADER)}"
                )
            header_seen = True
            continue
        if len(cells) != len(EVENTS_HEADER):
            raise AssertionError(
                f"{source}:{no}: registry row has {len(cells)} cells, schema owns {len(EVENTS_HEADER)}"
            )
        eid = cells[0]
        if not HEX8.fullmatch(eid):
            raise AssertionError(f"{source}:{no}: registry event-id {eid!r} is not 8-hex")
        if eid in seen:
            raise AssertionError(
                f"{source}:{no}: duplicate event-id {eid} in ## EVENTS registry "
                f"(first registered at line {seen[eid]})"
            )
        seen[eid] = no


def _events_table_row_spans(state_text: str) -> frozenset[tuple[int, int]]:
    """Char spans of registry-SHAPE rows strictly inside STATE.md's ## EVENTS.

    Section-scoped exemption (senior ruling c): only lines between the
    ``## EVENTS`` heading and the next ``##`` heading that match the normative
    row shape are exempt from 8-hex scanning; all other hex-bearing STATE.md
    lines stay in scan.
    """
    spans: set[tuple[int, int]] = set()
    in_events = False
    offset = 0
    for line in state_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("#"):
            in_events = stripped.strip() == "## EVENTS"
        elif in_events and REGISTRY_ROW_SHAPE.match(stripped):
            spans.add((offset, offset + len(line)))
        offset += len(line)
    return frozenset(spans)


def probe_hex_tokens(
    text: str,
    resolver: Callable[[str], bool],
    *,
    window: int = 80,
    source: str = "<text>",
    event_registry: frozenset[str] | None = None,
    exempt_spans: frozenset[tuple[int, int]] = frozenset(),
) -> None:
    """Every 8-hex token resolves as a revision or is a declared agent ID.

    F4′ third namespace: an occurrence introduced by a FULL-LINE
    ``EVENT: issues/<n>#issuecomment-<m> event-id <8hex>`` line never resolves
    via git and role vocabulary grants NO escape — it is valid only when listed
    in ``event_registry`` (the STATE.md ``## EVENTS`` resolution table).

    ``exempt_spans`` (section-scoped STATE.md exemption): occurrences whose
    span lies inside one are skipped outright — used for registry rows between
    the ``## EVENTS`` heading and the next ``##`` heading.
    """
    registry = event_registry if event_registry is not None else frozenset()
    event_spans = {m.span("eid") for m in EVENT_LINE.finditer(text)}
    for match in HEX8.finditer(text):
        token = match.group(0)
        if any(s <= match.start() and match.end() <= e for s, e in exempt_spans):
            continue
        if match.span() in event_spans:
            if token not in registry:
                raise AssertionError(
                    f"{source}: unregistered event-id {token} — EVENT-line tokens need a "
                    f"resolution row in the STATE.md ## EVENTS registry (role vocabulary "
                    f"is no escape in this namespace)"
                )
            continue
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
LEDGERS = ["agents/ledger/FIXES.md", "agents/ledger/DECISIONS.md", "agents/ledger/STATE.md"]
CONTRACT_DOCS = ["agents/contracts/MAIN_AGENT_CONTRACT.md", "agents/contracts/WORKER_CONTRACT.md", "agents/contracts/CONTRACT_TEMPLATE.md"]


def test_probe_a_live_ledger_tables_wellformed() -> None:
    for name in LEDGERS:
        probe_ledger_tables((ROOT / name).read_text(encoding="utf-8"), source=name)


def test_probe_b_live_contract_paths_resolve() -> None:
    for name in CONTRACT_DOCS:
        probe_backticked_paths((ROOT / name).read_text(encoding="utf-8"), ROOT, source=name)


def test_probe_c_live_8hex_tokens_declared_or_resolvable() -> None:
    events = _registered_event_ids()
    for name in ["agents/ledger/FIXES.md", "agents/ledger/DECISIONS.md"]:
        probe_hex_tokens(
            (ROOT / name).read_text(encoding="utf-8"),
            _git_resolves,
            source=name,
            event_registry=events,
        )
    # Senior ruling c: STATE.md joins the scan. Its ## EVENTS registry rows
    # are exempt SECTION-SCOPED (shape-matched, heading-bounded); every other
    # hex-bearing line stays in scan; the registry itself must satisfy the
    # normative schema with unique event-ids.
    state = "agents/ledger/STATE.md"
    state_text = (ROOT / state).read_text(encoding="utf-8")
    probe_event_registry_schema(state_text, source=state)
    probe_hex_tokens(
        state_text,
        _git_resolves,
        source=state,
        event_registry=events,
        exempt_spans=_events_table_row_spans(state_text),
    )


def test_probe_d_live_floor_quotes_match_gate_script() -> None:
    docs = {name: (ROOT / "agents/ledger" / name).read_text(encoding="utf-8") for name in ["SKLEARN_PIPELINES.md"]}
    docs["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
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

    seeded = _seeded_copy(tmp_path, "agents/ledger/FIXES.md", steal_pipe)
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

    seeded = _seeded_copy(tmp_path, "agents/contracts/MAIN_AGENT_CONTRACT.md", rename_dataflow)
    with pytest.raises(AssertionError, match="does not resolve in-tree"):
        probe_backticked_paths(seeded, ROOT, source="seeded-MAC")


def test_probe_c_red_unattributed_8hex_token(tmp_path: Path) -> None:
    # Hermetic seed: a live-ledger append would sit near real role words
    # (vocab window is ±80 chars), letting an unknown token pass via
    # neighboring declarations. Minimal string keeps the negative case pure.
    seeded_text = "Mystery reference cafe1234 ends here."

    def everything_except_seed(token: str) -> bool:
        return token != "cafe1234"  # deterministic stub resolver, no git dependence

    with pytest.raises(AssertionError, match="declared agent-ID namespace"):
        probe_hex_tokens(seeded_text, everything_except_seed, source="seeded-DECISIONS")


_FORGED_EVENT_LINE = "EVENT: issues/4#issuecomment-12345678 event-id deadbeef"


def _append_forged_event(lines: list[str]) -> list[str]:
    lines += [
        "",
        "Ruled by the senior reviewer agent synthesis panel:",
        _FORGED_EVENT_LINE,
    ]
    return lines


def test_probe_c_red_unregistered_event_id_amid_role_vocab(tmp_path: Path) -> None:
    # F4′ bypass attempt: an EVENT-line token fabricated amid role vocabulary.
    # The ±80-char vocab window is NO escape in this namespace — only a
    # ## EVENTS resolution row in STATE.md legitimizes the token.
    seeded = _seeded_copy(tmp_path, "agents/ledger/DECISIONS.md", _append_forged_event)
    at = seeded.index("deadbeef")
    assert ROLE_VOCABULARY.search(seeded[max(0, at - 80):at])  # vocab window IS lit
    with pytest.raises(AssertionError, match="unregistered event-id"):
        probe_hex_tokens(
            seeded, _git_resolves, source="seeded-DECISIONS",
            event_registry=_registered_event_ids(),
        )


def test_probe_c_green_registered_event_id_via_events_registry(tmp_path: Path) -> None:
    # Same seed as the RED twin; registering the event-id flips it GREEN —
    # proving registry membership, not vocabulary proximity, is the gate.
    seeded = _seeded_copy(tmp_path, "agents/ledger/DECISIONS.md", _append_forged_event)
    synthetic_state = (
        "## EVENTS\n"
        "| event-id | issue | comment-id | created_at | type | supersedes |\n"
        "|---|---|---|---|---|---|\n"
        "| deadbeef | issues/4#issuecomment-12345678 | 12345678 "
        "| 2026-08-24T16:19:00Z | amendment | - |\n"
    )
    assert _parse_event_registry(synthetic_state) == {"deadbeef"}  # parser round-trip
    registry = _registered_event_ids() | {"deadbeef"}  # live rows + the new one
    probe_hex_tokens(seeded, _git_resolves, source="seeded-DECISIONS", event_registry=registry)


def test_probe_c_red_duplicate_event_registry_row() -> None:
    # Normative-schema uniqueness constraint: the same hex8 registered twice
    # in ## EVENTS is RED even though each row individually is well-formed.
    duplicated_state = (
        "# STATE\n"
        "## EVENTS\n"
        "| event-id | issue | comment-id | created_at | type | supersedes |\n"
        "|---|---|---|---|---|---|\n"
        "| deadbeef | issues/4#issuecomment-12345678 | 12345678 "
        "| 2026-08-24T16:19:00Z | amendment | - |\n"
        "| cafe1234 | issues/3#issuecomment-5398091966 | 5398091966 "
        "| 2026-08-24T16:19:07Z | authorization | - |\n"
        "| deadbeef | issues/4#issuecomment-99999999 | 99999999 "
        "| 2026-08-24T16:20:00Z | verdict | - |\n"
        "## Next Section\n"
    )
    with pytest.raises(AssertionError, match="duplicate event-id deadbeef"):
        probe_event_registry_schema(duplicated_state, source="seeded-STATE")


def test_probe_c_event_grammar_canonical_bytes_pinned() -> None:
    # 8428b3e lesson: ruled wording and detector must not drift. The compiled
    # EVENT_LINE pattern IS the senior's canonical grammar bytes, and those
    # bytes appear verbatim in this module's source (no silent retyping).
    assert EVENT_LINE.pattern == EVENT_GRAMMAR_CANONICAL
    module_source = Path(__file__).read_text(encoding="utf-8")
    assert EVENT_GRAMMAR_CANONICAL in module_source


def test_probe_d_red_stale_floor_quote(tmp_path: Path) -> None:
    script = (ROOT / "scripts/run_local_ci.sh").read_text(encoding="utf-8")

    def stale_floor(lines: list[str]) -> list[str]:
        return [ln.replace("cov-fail-under=95", "cov-fail-under=94") for ln in lines]

    seeded = _seeded_copy(tmp_path, "agents/ledger/SKLEARN_PIPELINES.md", stale_floor)
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
