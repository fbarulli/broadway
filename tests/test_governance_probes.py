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
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, date, datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

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


# --------------------------------------------------------------------------- #
# Stamp-semantics tripwire (doctrine text lives in MAC_APPENDIX.md; the
# MAIN_AGENT_CONTRACT.md §5 bullet is a pointer there since D32(6)):
# standing contracts must
# never encode absolute-SHA equality as an executable precondition ("must equal
# <sha>"). Dispatch stamps are relative by law; absolute SHAs belong only to
# immutable records as provenance anchors. G0B.md is the single frozen
# completed-dispatch exception (archival note pending a separate decision) —
# the baseline below pins it, so any NEW occurrence anywhere under
# agents/contracts/ turns this probe RED.
SHA_GATE_LINE = re.compile(
    r"rev-parse[^\n]*must\s+equal[^\n]*[0-9a-f]{6,40}",
    re.IGNORECASE,
)
SHA_GATE_BASELINE: dict[str, int] = {"G0B.md": 1}


def test_no_new_absolute_sha_gates_in_contracts() -> None:
    hits: dict[str, int] = {}
    for path in sorted((ROOT / "agents" / "contracts").glob("*.md")):
        count = len(SHA_GATE_LINE.findall(path.read_text(encoding="utf-8")))
        if count:
            hits[path.name] = count
    assert hits == SHA_GATE_BASELINE, (
        f"absolute-sha equality gates changed under agents/contracts/: {hits} "
        f"(baseline {SHA_GATE_BASELINE}). Dispatch stamps must stay relative — "
        "see MAC_APPENDIX.md Stamp semantics "
        "(pointer at MAIN_AGENT_CONTRACT.md §5)."
    )


# --------------------------------------------------------------------------- #
# Probe (e): new-surface tripwire (capture mechanism #2; human ruling under
# D31): an unregistered TRACKED file under a governed code-bearing surface is
# RED. STRICT DECLARATION GRAMMAR (post-vacuity repair, adversarial-review
# mandated): scanning every row string let prose slash-tokens blanket-declare
# all six roots, so the live half could never fire. Declarations are read
# ONLY from inputs:/outputs: list entries and the path prefix of owner:
# strings — NEVER from transforms:/findings:/touched_by:/validated_by:/
# if_changed: free text — and directory/glob declarations must name a
# SUBSURFACE (>=2 path segments): a bare 'src/' mid-sentence grants nothing,
# while 'experiments/results/' and 'k8s/optuna/**' register their subtree.
# Pure classification lives in the functions below; the single test node
# carries the LIVE half plus in-process falsifiability proofs against the
# LIVE registry shape (no git writes, no tree mutation — fixture law).
# --------------------------------------------------------------------------- #
TRIPWIRE_SURFACES = ("src", "scripts", "k8s", "experiments", "project", "tests")
BRACED_PATH = re.compile(r"((?:[\w.-]+/)+)\{([^{}]+)\}([\w./-]*)")
SURFACE_PATH_TOKEN = re.compile(r"(?:[\w.-]+/)+[\w.*-]*")
OWNER_LINE_REF = re.compile(r":\d[\d-]*$")
NEW_SURFACE_EXEMPTS: dict[str, str] = {
    "project/scripts": "D32(4) intentionally record-free teaching surfaces "
                       "(MAIN_AGENT_CONTRACT.md §14 hygiene bullet; "
                       "00-resolutions.md §6 addendum)",
}
# (path-or-prefix, reason, review-date ISO) — path = a file or a directory
# prefix (trailing slash optional); entries past their review date fail loud
# ('allowlist entry expired'). Seeded from the strict-grammar sweep at
# meta.head=1cf33b5: the four prefixes below are genuinely uncovered because
# their ownership prose lives in grammar-excluded fields.
NEW_SURFACE_ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    ("experiments/multivariate/",
     ("family custody lives only in GATE-INFRA-129 transform prose "
      "(SPLIT-OWNER sublanes), excluded from declarations; promote to an "
      "inputs/outputs entry or dedicated row"), "2026-09-08"),
    ("experiments/polynomial_regression_et_all/",
     ("same GATE-INFRA-129 transform-prose custody, excluded from "
      "declarations; promote to an inputs/outputs entry"), "2026-09-08"),
    ("experiments/univariate/fare_amount_trip_distance/",
     ("subdir named only in excluded free text; parent "
      "experiments/univariate remains registry-covered"), "2026-09-08"),
    ("project/tests/",
     ("ingest-gate pins ride findings/validated_by fields, excluded from "
      "declarations; promote the pins to inputs/outputs"), "2026-09-08"),
)


def _today() -> date:
    """Current date, timezone-aware per repo convention (DTZ011-clean)."""
    return datetime.now(tz=UTC).date()


def _surface_tokens(text: str) -> Iterator[tuple[str, bool]]:
    """Surface-rooted tokens as (path, declares_subsurface) — strict forms.

    Directory power (trailing-slash and glob tokens) requires >=2 path
    segments: a bare surface-root token names the governed tree itself, it
    does not declare a registrable subsurface. Exact-file form requires a
    basename extension (Dockerfile-style names included); an extensionless
    path reference declares nothing.
    """
    expanded = BRACED_PATH.sub(
        lambda m: " ".join(m.group(1) + alt.strip() + m.group(3)
                           for alt in m.group(2).split(",")),
        text,
    )
    for match in SURFACE_PATH_TOKEN.finditer(expanded):
        token = match.group(0).rstrip(".,")
        if token.split("/", 1)[0] not in TRIPWIRE_SURFACES:
            continue
        if "*" in token:
            base = token.split("*", 1)[0].rstrip("/")
        elif token.endswith("/"):
            base = token.rstrip("/")
        elif "." in token.rsplit("/", 1)[-1]:
            yield token, False  # exact-file form: basename bears an extension
            continue
        else:
            continue  # extensionless path reference declares nothing
        if "/" in base:
            yield base, True


def _declaration_texts(row: dict[str, object]) -> Iterator[str]:
    """Strings allowed to declare coverage: owner path prefix, inputs, outputs.

    All gates.yaml rows carry these three keys (verified across the registry);
    direct indexing fails loud on schema drift. transforms/findings/
    touched_by/validated_by/if_changed free text is deliberately never read.
    """
    head = str(row["owner"]).split(None, 1)
    yield OWNER_LINE_REF.sub("", head[0].rstrip(":")) if head else ""
    for entry in [*(row["inputs"] or []), *(row["outputs"] or [])]:
        yield str(entry)


class SurfaceCoverage(NamedTuple):
    """gates.yaml-derived registration prefixes for governed surface files."""

    declared_dirs: frozenset[str]
    mentioned_dirs: frozenset[str]
    files: frozenset[str]


def parse_surface_coverage(rows: list[dict[str, object]], root: Path) -> SurfaceCoverage:
    """Collapse strict-grammar declaration texts into the three tiers."""
    declared: set[str] = set()
    mentioned: set[str] = set()
    files: set[str] = set()
    for row in rows:
        for value in _declaration_texts(row):
            for path, declares_subsurface in _surface_tokens(value):
                if declares_subsurface or (root / path).is_dir():
                    declared.add(path)
                else:
                    files.add(path)
                    mentioned.add(path.rsplit("/", 1)[0])
    return SurfaceCoverage(frozenset(declared), frozenset(mentioned), frozenset(files))


def find_unregistered(
    tracked: Sequence[str], coverage: SurfaceCoverage, exemptions: dict[str, str],
) -> list[str]:
    """Tracked surface files covered by NO tier and NO cited exemption."""

    def registered(path: str) -> bool:
        segments = path.split("/")
        ancestors = ["/".join(segments[:i]) for i in range(1, len(segments))]
        if not ancestors:
            return False
        parent = ancestors[-1]
        if path in coverage.files:
            return True
        if parent in coverage.declared_dirs or parent in coverage.mentioned_dirs:
            return True
        return any(dir_ in coverage.declared_dirs for dir_ in ancestors[:-1])

    def exempted(path: str) -> bool:
        for prefix in exemptions:  # trailing slash on a prefix is tolerated
            bare = prefix.rstrip("/")
            if path == bare or path.startswith(bare + "/"):
                return True
        return False

    return sorted(
        p for p in tracked
        if p.split("/", 1)[0] in TRIPWIRE_SURFACES
        and not registered(p)
        and not exempted(p)
    )


def expired_allowlist_entries(
    allowlist: Sequence[tuple[str, str, str]], today: date,
) -> list[str]:
    """Allowlist (path, reason, review-date) rows strictly past their review."""
    return [
        f"{path} ({reason}; review {review})"
        for path, reason, review in allowlist
        if date.fromisoformat(review) < today
    ]


def probe_new_surfaces_registered(
    tracked: Sequence[str],
    registry_rows: list[dict[str, object]],
    *,
    today: date,
    root: Path = ROOT,
    exemptions: dict[str, str] | None = None,
    allowlist: Sequence[tuple[str, str, str]] = NEW_SURFACE_ALLOWLIST,
    source: str = "agents/ledger/gates.yaml",
) -> None:
    """New-surface tripwire: an unregistered tracked surface file is RED.

    Human ruling (eight-packet batch follow-up, capture mechanism #2), under
    D31's REGISTRY-AUDIT duty. SURFACES are the six code-bearing top-level
    trees in ``TRIPWIRE_SURFACES``; ``.venv``/``data``/``artifacts`` are
    excluded by construction and docs/ledger/contracts trees are exempt
    outright — ``agents/**`` carries its own ownership law
    (HELPER_FILE_OWNERSHIP.md) and self-describes through probes a-d.

    Registration is derived from ``agents/ledger/gates.yaml`` rows ONLY
    (never ``meta:`` — the ledger may not self-register code), under a STRICT
    declaration grammar: declaration-bearing text is read ONLY from
    ``inputs:``/``outputs:`` list entries and the path prefix (first
    whitespace token, line-reference stripped) of ``owner:`` — NEVER from
    ``transforms:``/``findings:``/``touched_by:``/``validated_by:``/
    ``if_changed:`` free text. Within those fields only unambiguous forms
    count: exact file paths bearing a basename extension (Dockerfile-style
    names included), trailing-slash directory tokens, ``k8s/optuna/
    **``-style globs (prefix before the first ``*``), and brace-expanded
    ``experiments/{a,b}/`` lists; a yielded token that nonetheless resolves
    to a live tree directory lands in the declared tier. Directory power additionally requires a SUBSURFACE
    (>=2 path segments): a bare root token ('src/', 'tests/') mid-entry names
    the governed tree itself and declares nothing. Three coverage tiers —
    (files) a declaration names the path itself; (parent) a declaration names
    a path whose immediate parent directory equals the file's directory, with
    NO further cascade; (declared) an ANCESTOR subsurface was declared.
    Parent-tier mentions never cascade upward: one stray file mention must
    not blanket a whole surface, or the tripwire could never fire.

    Vacuity incident: an earlier draft scanned EVERY row string, so ~13 rows'
    prose slash-tokens blanket-declared all six governed roots through the
    declared tier and no synthetic ghost could ever fire; this grammar is the
    mandated minimum repair.

    ``project/scripts/*`` is EXEMPT BY NAME with citation — D32(4) rules these
    intentionally record-free teaching surfaces (MAIN_AGENT_CONTRACT.md §14
    hygiene bullet + 00-resolutions.md §6 addendum) — deliberately not an ad
    hoc allowlist row. ``NEW_SURFACE_ALLOWLIST`` is the dated escape hatch;
    entries strictly past their review date fail loud (decay by design).
    Seeded NON-EMPTY from the strict sweep at meta.head=1cf33b5: exactly four
    genuinely uncovered prefixes remain (experiments/multivariate/,
    experiments/polynomial_regression_et_all/,
    experiments/univariate/fare_amount_trip_distance/, project/tests/) because
    their ownership lives in grammar-excluded fields; each seed states its
    promotion path and reviews 2026-09-08. The experiments/more_modeling
    batch stays registered by GATE-INFRA-93's amended parity input string.
    """
    exemptions = NEW_SURFACE_EXEMPTS if exemptions is None else dict(exemptions)
    failures = [
        f"allowlist entry expired: {entry}"
        for entry in expired_allowlist_entries(allowlist, today)
    ]
    # Active (unexpired) allowlist entries ARE registration: subtract them.
    effective = {
        **exemptions,
        **{path: f"allowlisted: {reason}"
           for path, reason, review in allowlist
           if date.fromisoformat(review) >= today},
    }
    unregistered = find_unregistered(
        list(tracked), parse_surface_coverage(registry_rows, root), effective,
    )
    if unregistered:
        failures.append(
            "unregistered tracked file(s) under governed surfaces "
            f"{list(TRIPWIRE_SURFACES)}: {unregistered} — add a gates.yaml row "
            "naming the path/directory, a NEW_SURFACE_EXEMPTS citation, or a "
            "dated NEW_SURFACE_ALLOWLIST entry"
        )
    if failures:
        raise AssertionError(f"{source}: " + "; ".join(failures))


def test_probe_e_new_surface_tripwire_live_and_falsifiable() -> None:
    """LIVE green at HEAD; per-root ghosts fire against the LIVE registry."""
    doc = yaml.safe_load((ROOT / "agents/ledger/gates.yaml").read_text(encoding="utf-8"))
    live_rows: list[dict[str, object]] = doc["gates"]
    tracked = subprocess.run(
        ["git", "ls-files", "--", *TRIPWIRE_SURFACES],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.splitlines()
    coverage = parse_surface_coverage(live_rows, ROOT)
    probe_new_surfaces_registered(tracked, live_rows, today=_today())
    # Six-ghost negative controls — one synthetic file under EACH governed
    # root, judged against the FULL LIVE registry shape (nothing stripped):
    # each must be classified uncovered and each must turn the probe RED.
    for root_name in TRIPWIRE_SURFACES:
        ghost = f"{root_name}/zzz_ghost_probe/{root_name}_ghost.py"
        assert find_unregistered([ghost], coverage, NEW_SURFACE_EXEMPTS) == [ghost]
        with pytest.raises(AssertionError, match="unregistered tracked file"):
            probe_new_surfaces_registered([ghost], live_rows, today=_today())
    # Blanket-transform control — the adversarial-review sentence declaring
    # two roots mid-prose registers NOTHING under the strict grammar: a
    # transforms-only row yields an empty SurfaceCoverage, and even inside a
    # realistic full row (owner/inputs/outputs present) the ghost stays red.
    blanket_sentence = "zero callers in src/, tests integrated via scripts/"
    prose_only_cov = parse_surface_coverage([{
        "id": "SYNTH-PROSE", "phase": "synth", "order": 0, "owner": "",
        "inputs": [], "outputs": [], "transforms": [blanket_sentence],
    }], ROOT)
    assert prose_only_cov == SurfaceCoverage(frozenset(), frozenset(), frozenset())
    blanket_row = {
        "id": "SYNTH-BLANKET", "phase": "synth", "order": 0,
        "owner": "src/broadway/ok_module.py:1 run()",
        "inputs": [], "outputs": [],
        "transforms": [blanket_sentence],
    }
    blanket_cov = parse_surface_coverage([blanket_row], ROOT)
    assert not blanket_cov.declared_dirs  # the prose sentence granted nothing
    blanket_ghost = "src/zzz_ghost_probe/src_ghost.py"
    assert find_unregistered(
        [blanket_ghost], blanket_cov, NEW_SURFACE_EXEMPTS,
    ) == [blanket_ghost]
    with pytest.raises(AssertionError, match="unregistered tracked file"):
        probe_new_surfaces_registered(
            [blanket_ghost], [blanket_row], today=_today(),
        )
    # Rollback proof — strike every row whose ACCEPTED fields mention the
    # experiments tree: with registration collapsed, the tracked batch fires.
    survivors = [r for r in live_rows
                 if not any("experiments" in t for t in _declaration_texts(r))]
    assert survivors != live_rows  # real rows do declare the tree
    batch = [p for p in tracked if p.startswith("experiments/")]
    assert batch
    with pytest.raises(AssertionError, match="unregistered tracked file"):
        probe_new_surfaces_registered(
            [batch[0]], survivors, today=_today(),
        )
    # Decay proof — an expired entry fails loud; the same entry unexpired
    # registers an otherwise-uncovered path.
    stale = (("experiments/legacy_one_off.py", "intentional one-off", "2026-01-01"),)
    fresh = (("experiments/legacy_one_off.py", "intentional one-off", "2999-01-01"),)
    with pytest.raises(AssertionError, match="allowlist entry expired"):
        probe_new_surfaces_registered(
            ["experiments/legacy_one_off.py"], [], today=_today(), allowlist=stale,
        )
    probe_new_surfaces_registered(
        ["experiments/legacy_one_off.py"], [], today=_today(), allowlist=fresh,
    )
    # Exemption proof — project/scripts passes BY CITATION, not by allowlist.
    probe_new_surfaces_registered(
        ["project/scripts/99_future_teaching.py"], [], today=_today(),
    )
