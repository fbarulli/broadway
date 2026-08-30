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
from collections.abc import Callable, Iterator, Mapping, Sequence
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
                 "configs/experiment/baseline.yaml", "k8s/deploy.yaml", "tools/run.sh", "uv.lock"]:
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
# Stamp-semantics tripwire (doctrine text lives IN MAIN_AGENT_CONTRACT.md §5
# since the 2026-08-26 D37 retirement of MAC_APPENDIX.md):
# standing contracts must
# never encode absolute-SHA equality as an executable precondition ("must equal
# <sha>"). Dispatch stamps are relative by law; absolute SHAs belong only to
# immutable records as provenance anchors. The single frozen completed-
# dispatch exception (G0B.md) was retired WITH its file on 2026-08-26 —
# the empty baseline below means ANY occurrence anywhere under
# agents/contracts/ turns this probe RED.
SHA_GATE_LINE = re.compile(
    r"rev-parse[^\n]*must\s+equal[^\n]*[0-9a-f]{6,40}",
    re.IGNORECASE,
)
SHA_GATE_BASELINE: dict[str, int] = {}


def test_no_new_absolute_sha_gates_in_contracts() -> None:
    hits: dict[str, int] = {}
    for path in sorted((ROOT / "agents" / "contracts").glob("*.md")):
        count = len(SHA_GATE_LINE.findall(path.read_text(encoding="utf-8")))
        if count:
            hits[path.name] = count
    assert hits == SHA_GATE_BASELINE, (
        f"absolute-sha equality gates changed under agents/contracts/: {hits} "
        f"(baseline {SHA_GATE_BASELINE}). Dispatch stamps must stay relative — "
        "see MAIN_AGENT_CONTRACT.md §5 Stamp semantics."
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
# while 'project/experiments/results/' and 'k8s/optuna/**' register their subtree.
# Pure classification lives in the functions below; the single test node
# carries the LIVE half plus in-process falsifiability proofs against the
# LIVE registry shape (no git writes, no tree mutation — fixture law).
# --------------------------------------------------------------------------- #
TRIPWIRE_SURFACES = ("src", "scripts", "k8s", "project", "tests")
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
# ('allowlist entry expired'). The strict-grammar sweep is fully represented
# by accepted registry declarations; temporary exemptions are empty.
NEW_SURFACE_ALLOWLIST: tuple[tuple[str, str, str], ...] = ()


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
    ``project/experiments/{a,b}/`` lists; a yielded token that nonetheless resolves
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
    The strict-grammar sweep is represented by accepted registry declarations:
    GATE-INFRA-129 names the experiment subtrees and GATE-INFRA-92 names
    project/tests/**. Temporary allowlist entries are therefore empty.
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
        ["git", "ls-files", "-co", "--exclude-standard", "--", *TRIPWIRE_SURFACES],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.splitlines()
    tracked = [path for path in tracked if (ROOT / path).is_file()]
    coverage = parse_surface_coverage(live_rows, ROOT)
    probe_new_surfaces_registered(tracked, live_rows, today=_today())
    # Five-ghost negative controls — one synthetic file under EACH governed
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
    batch = [p for p in tracked if p.startswith("project/experiments/")]
    assert batch
    with pytest.raises(AssertionError, match="unregistered tracked file"):
        probe_new_surfaces_registered(
            [batch[0]], survivors, today=_today(),
        )
    # Decay proof — an expired entry fails loud; the same entry unexpired
    # registers an otherwise-uncovered path.
    stale = (("project/experiments/legacy_one_off.py", "intentional one-off", "2026-01-01"),)
    fresh = (("project/experiments/legacy_one_off.py", "intentional one-off", "2999-01-01"),)
    with pytest.raises(AssertionError, match="allowlist entry expired"):
        probe_new_surfaces_registered(
            ["project/experiments/legacy_one_off.py"], [], today=_today(), allowlist=stale,
        )
    probe_new_surfaces_registered(
        ["project/experiments/legacy_one_off.py"], [], today=_today(), allowlist=fresh,
    )
    # Exemption proof — project/scripts passes BY CITATION, not by allowlist.
    probe_new_surfaces_registered(
        ["project/scripts/99_future_teaching.py"], [], today=_today(),
    )


# --------------------------------------------------------------------------- #
# Probes f-j — enforcement lane WH (five teeth from Scout-4's findings).
# House pattern as probes a-e: constants + pure helpers + ONE test node each,
# carrying the LIVE half plus in-test falsifiability against synthetic
# corruption (no tree mutation — fixture law). Datetimes ride _today()
# (DTZ-clean); git access is read-only inspection per WORKER_CONTRACT.
#
# Rulings landed with this lane (human-sanctioned custody extensions):
# gates.yaml meta.head re-stamped 1cf33b5->f638f27 (probe f found the stamp
# in violation of the parent-stamp law it pins), SENIOR.md:111 rewritten to
# the sanctioned cache-root form (probe i's live contradiction).
# --------------------------------------------------------------------------- #
GATES_REGISTRY = "agents/ledger/gates.yaml"
STATE_LEDGER = "agents/ledger/STATE.md"
CI_WORKFLOW = ".github/workflows/ci.yml"


# --- Probe (f): meta.head parent-stamp pin -------------------------------- #
# Law derived from precedent (abcb729 stamped its parent; b06bdd2-era held
# too): meta.head records the FIRST PARENT of the last commit touching
# gates.yaml. A registry edit must re-stamp; anything else drifts.
def parse_meta_head(registry_text: str, source: str = GATES_REGISTRY) -> str:
    """The short-sha stamp under the gate registry's meta: block."""
    try:
        document = yaml.safe_load(registry_text)
    except yaml.YAMLError as error:
        raise AssertionError(f"{source}: invalid YAML registry") from error
    meta = document.get("meta") if isinstance(document, Mapping) else None
    if not isinstance(meta, Mapping) or not isinstance(meta.get("head"), str):
        raise AssertionError(f"{source}: no `head:` stamp under the meta: block")
    return meta["head"]


def assert_parent_stamp(actual: str, required: str, source: str = GATES_REGISTRY) -> None:
    """Pure comparison: meta.head must equal the toucher's first parent."""
    if actual != required:
        raise AssertionError(
            f"{source}: meta.head {actual!r} violates the parent-stamp law — required "
            f"{required!r} (first parent of the last commit touching {GATES_REGISTRY}); "
            "re-stamp meta.head with every registry edit (precedent abcb729)"
        )


def required_meta_head(parent: str, head: str, dirty: bool) -> str:
    """Dirty registry content is stamped for its prospective parent."""
    return head if dirty else parent


def registry_is_dirty(path: str) -> bool:
    """Use HEAD as the comparison point for a registry edit."""
    return bool(subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT, check=False,
    ).returncode)


@cache
def _last_commit_touching(path: str) -> str | None:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", path],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    return proc.stdout.strip() or None


def _first_parent(commit: str) -> str | None:
    """Short hash of ``commit``'s first parent; None for a root commit.

    rev-list --parents -n 1 prints '<commit>' ALONE when no parent exists —
    explicit root detection instead of parsing rev-parse failures.
    """
    out = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", commit],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.split()
    if len(out) == 1:
        return None
    return subprocess.run(
        ["git", "rev-parse", "--short=7", out[1]],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.strip()


def test_probe_f_meta_head_parent_stamp_live_and_falsifiable() -> None:
    """LIVE green at HEAD; wrong synthetic stamps fire through the pure helper."""
    toucher = _last_commit_touching(GATES_REGISTRY)
    assert toucher is not None, f"{GATES_REGISTRY} has no committing history?"
    actual = parse_meta_head((ROOT / GATES_REGISTRY).read_text(encoding="utf-8"))
    parent = _first_parent(toucher)
    if parent is None:
        # Root-commit edge: nothing precedes the toucher, no stamp can exist.
        print(f"[P1] SKIP-WITH-PASS: {toucher} ({GATES_REGISTRY} toucher) is the root commit")
        pytest.skip("gates.yaml toucher is the root commit — no parent stamp derivable")
    dirty = registry_is_dirty(GATES_REGISTRY)
    head = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.strip()
    required = required_meta_head(parent, head, dirty)
    print(f"[P1] meta.head parent-stamp: dirty={dirty} required={required} actual={actual} match={actual == required}")
    assert_parent_stamp(actual, required)
    # Negative control — a synthetic wrong stamp through the pure helper.
    with pytest.raises(AssertionError, match="violates the parent-stamp law"):
        assert_parent_stamp("0badf00d", required, source="synthetic-gates")
    assert_parent_stamp(required, required, source="synthetic-gates")  # positive twin
    assert required_meta_head(parent, head, True) == head
    with pytest.raises(AssertionError, match="no `head:` stamp"):  # parser fails loud
        parse_meta_head("meta:\n  assembled: '2026-08-24'\n", source="synthetic-gates")
    assert parse_meta_head("x:\n  head: wrong\nmeta:\n  head: right\n") == "right"


# --- Probe (g): EVENTS tamper-lock (D31(2)) ------------------------------- #
# A change to STATE.md's normative ## EVENTS table between two commits needs
# an authorizing instrument: the landing commit body must cite >=1 event-id
# holding a row in the POST-change registry. Identical sections pass free.
def events_section(state_text: str) -> str:
    """STATE.md bytes between ## EVENTS and the next top-level ## heading."""
    lines: list[str] = []
    in_events = False
    for line in state_text.splitlines(keepends=True):
        if line.startswith("#"):
            if in_events:
                break
            in_events = line.strip() == "## EVENTS"
            continue
        if in_events:
            lines.append(line)
    return "".join(lines)


def normalized_ws(text: str) -> str:
    """Whitespace-collapsed form for tamper-insensitive comparison."""
    return " ".join(text.split())


def assert_events_change_authorized(
    before: str,
    after: str,
    commit_body: str,
    post_table_ids: frozenset[str] | set[str],
) -> None:
    """Tamper-lock over the ## EVENTS table (pure; feeds the negative control)."""
    if normalized_ws(before) == normalized_ws(after):
        return
    cited = sorted({t for t in HEX8.findall(commit_body) if t in post_table_ids})
    if not cited:
        raise AssertionError(
            "STATE.md ## EVENTS changed WITHOUT authorization: the landing commit "
            "body cites none of the POST-change registry event-ids (D31(2) requires "
            "an authorizing event row); hex tokens seen in body: "
            f"{sorted(set(HEX8.findall(commit_body)))}"
        )


def test_probe_g_events_tamper_lock_live_and_falsifiable() -> None:
    """LIVE: HEAD vs working-tree EVENTS agree today; tampering goes RED."""
    head_text = subprocess.run(
        ["git", "show", f"HEAD:{STATE_LEDGER}"],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    work_text = (ROOT / STATE_LEDGER).read_text(encoding="utf-8")
    body = subprocess.run(
        ["git", "log", "-1", "--format=%B"],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    # Falsifiability first, so a live RED below can only mean the TREE, never
    # an unproven detector: negative controls on synthetic strings...
    with pytest.raises(AssertionError, match="WITHOUT authorization"):
        assert_events_change_authorized("a | b", "a | c", "routine refresh cafe1234", {"deadbeef"})
    assert_events_change_authorized(  # authorized twin: body cites the row id
        "a | b", "a | c", "board ruling event deadbeef recorded", {"deadbeef"},
    )
    assert_events_change_authorized("x", "x\n ", "", set())  # identical always passes
    # ...then the clean-tree twin on REAL HEAD data (HEAD vs HEAD is always
    # authorized) proving the live comparison shape on actual registry bytes.
    assert_events_change_authorized(
        events_section(head_text), events_section(head_text),
        body, _parse_event_registry(work_text),
    )
    # LIVE verdict last: RED here means working-tree EVENTS diverged from HEAD
    # without an authorizing commit body (e.g. another lane's WIP rows).
    assert_events_change_authorized(
        events_section(head_text), events_section(work_text),
        body, _parse_event_registry(work_text),
    )


# --- Probe (h): repo-root dot-dir / tracked-cache ban (L5) ---------------- #
SANCTIONED_ROOT_DOT_DIRS = frozenset({
    ".git", ".github", ".mplconfig",  # floor: git itself, workflows, sanctioned MPL root
    ".venv", ".uv-cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
BANNED_TRACKED_PREFIXES = (".uv-cache/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/")


def stray_root_dot_dirs(entries: Sequence[str], allowed: frozenset[str]) -> list[str]:
    """Repo-root dot-DIRECTORIES outside the sanctioned set."""
    return sorted(e for e in entries if e.startswith(".") and e not in allowed)


def banned_tracked_paths(
    tracked: Sequence[str], prefixes: Sequence[str] = BANNED_TRACKED_PREFIXES,
) -> list[str]:
    """Tracked files living under a banned cache root."""
    return sorted(p for p in tracked if any(p.startswith(pre) for pre in prefixes))


def assert_root_hygiene(stray: Sequence[str], banned: Sequence[str], source: str = "repo-root") -> None:
    """Both ban classes RED together; empty inputs pass."""
    problems = []
    if stray:
        problems.append(f"unsanctioned dot-dir(s) at repo root: {list(stray)}")
    if banned:
        problems.append(f"tracked file(s) under banned cache roots: {list(banned)}")
    if problems:
        raise AssertionError(f"{source}: " + "; ".join(problems))


def test_probe_h_root_dot_dir_and_cache_ban_live_and_falsifiable() -> None:
    """LIVE green at HEAD; stray dot-dirs and tracked caches go RED."""
    entries = [p.name for p in ROOT.iterdir() if p.is_dir()]
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.splitlines()
    stray = stray_root_dot_dirs(entries, SANCTIONED_ROOT_DOT_DIRS)
    banned = banned_tracked_paths(tracked)
    print(f"[P3] root dot-dirs={[e for e in entries if e.startswith('.')]} banned-tracked={banned}")
    assert_root_hygiene(stray, banned)
    # Negative controls (synthetic): a new tool-cache dir at root...
    with pytest.raises(AssertionError, match="unsanctioned dot-dir"):
        assert_root_hygiene([".tox"], [])
    # ...a cache root sneaking into git tracking...
    with pytest.raises(AssertionError, match="banned cache roots"):
        assert_root_hygiene([], [".uv-cache/wheels/x.whl"])
    # ...and the clean twin passes.
    assert_root_hygiene([], [])


# --- Probe (i): UV cache wrapper ownership (L10, fail-loud fallback) ------ #
# Creation-time law (arbitration row 146 / D32): only scripts/uv.sh may select
# UV_CACHE_DIR. It must choose a host-local cache and fail loud if unavailable.
# Any other tracked mention outside dated historical records is RED.
UV_CACHE_NEEDLE = "UV_CACHE_DIR="
UV_CACHE_RUNTIME_OWNER = "scripts/uv.sh"
UV_CACHE_EXEMPTIONS: tuple[tuple[str, str, str], ...] = (
    ("agents/ledger/DECISIONS.md",
     "historical D28/D32 measured-run records", "2026-09-08"),
    ("agents/ledger/STATE.md",
     "courtesy D28 suite-tail baseline command", "2026-09-08"),
    ("agents/audits/B-TRUTH-ENFORCEMENT.md",
     "audit prose quoting an in-venv run observation", "2026-09-08"),
    ("agents/ledger/arbitration/2026-08-24/H-infra-custody-verdicts.md",
     "row-146 cache-root fork-law verdict prose", "2026-09-08"),
    ("agents/ledger/arbitration/2026-08-24/consolidation-slate.md",
     "GATE-INFRA-146 slate entry (historical proposal record)", "2026-09-08"),
    ("agents/ledger/arbitration/2026-08-24/gap-object-creators.md",
     "packet-H finding that created the fork-site law", "2026-09-08"),
    ("agents/ledger/gates.yaml",
     "HISTORY finding string documenting the retired ship.sh fork site", "2026-09-08"),
)


def parse_git_grep_hits(payload: str) -> list[tuple[str, int]]:
    """`git grep -n` rows into (path, lineno) pairs."""
    return [(path, int(no)) for path, no, _line in (r.split(":", 2) for r in payload.splitlines())]


# This module necessarily quotes the needle verbatim in its falsifiability
# fixtures and failure-message strings; the live scan skips itself by path.
UV_CACHE_SELF_SCAN_PATHS = frozenset({"tests/test_governance_probes.py"})


def unexempted_uv_cache_hits(
    hits: Sequence[tuple[str, int]],
    exemptions: Sequence[tuple[str, str, str]],
    today: date,
) -> list[str]:
    """Hits whose path carries no ACTIVE dated exemption."""
    active = {path for path, _reason, review in exemptions if date.fromisoformat(review) >= today}
    allowed = active | {UV_CACHE_RUNTIME_OWNER}
    return [f"{path}:{no}" for path, no in hits if path not in allowed]


def probe_uv_cache_export_ban(
    hits: Sequence[tuple[str, int]],
    *,
    today: date,
    exemptions: Sequence[tuple[str, str, str]] = UV_CACHE_EXEMPTIONS,
    source: str = "git grep UV_CACHE_DIR=",
) -> None:
    """Only the fail-loud wrapper may select a runtime uv cache."""
    failures = [
        f"exemption entry expired: {entry}"
        for entry in expired_allowlist_entries(exemptions, today)
    ]
    offenders = unexempted_uv_cache_hits(hits, exemptions, today)
    if offenders:
        failures.append(
            f"UV_CACHE_DIR= outside dated exemption records: {offenders} — "
            "only scripts/uv.sh may select the host-local cache; remove the override "
            "or add a dated UV_CACHE_EXEMPTIONS row with a stated reason"
        )
    if failures:
        raise AssertionError(f"{source}: " + "; ".join(failures))


def test_probe_i_uv_cache_wrapper_owner_live_and_falsifiable() -> None:
    """LIVE green; any cache selector outside the wrapper goes RED."""
    payload = subprocess.run(
        ["git", "grep", "-n", UV_CACHE_NEEDLE],
        capture_output=True, text=True, cwd=ROOT, check=False,
    ).stdout  # exit 1 == zero hits: the future-green shape parses to []
    hits = [
        (path, no) for path, no in parse_git_grep_hits(payload)
        if path not in UV_CACHE_SELF_SCAN_PATHS  # this module's own fixture quotes
    ]
    probe_uv_cache_export_ban(hits, today=_today())
    # The wrapper is the sole live selector.
    probe_uv_cache_export_ban([("scripts/uv.sh", 22)], today=_today())
    # Negative control grounded in this lane's own repair: a ship-path
    # override must classify RED...
    with pytest.raises(AssertionError, match="outside dated exemption"):
        probe_uv_cache_export_ban([("scripts/ship.sh", 10)], today=_today())
    # ...an exempted historical record stays GREEN...
    probe_uv_cache_export_ban([("agents/ledger/DECISIONS.md", 367)], today=_today())
    # ...an expired exemption fails loud...
    stale = (("agents/ledger/DECISIONS.md", "historical", "2026-01-01"),)
    with pytest.raises(AssertionError, match="exemption entry expired"):
        probe_uv_cache_export_ban([("agents/ledger/DECISIONS.md", 367)], today=_today(), exemptions=stale)
    # ...and the parser round-trips a realistic grep payload.
    sample = "scripts/ship.sh:10:export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.uv-cache}\n"
    assert parse_git_grep_hits(sample) == [("scripts/ship.sh", 10)]


# --- Probe (j): CI gate-divergence watcher (L2) --------------------------- #
# Same decay philosophy as probe e: TODAY'S ci.yml run-block commands are the
# frozen dated baseline; a future run-line whose command head is neither in
# the hand-written floor, the dated snapshot, nor the run_local_ci SSOT rule
# goes RED until deliberately baselined. Continuation lines (trailing `\`)
# join into their logical command so only NEW command heads trip.
RUN_LOCAL_CI_INVOCATION = "bash scripts/run_local_ci.sh"
TOKEN_FLOOR = frozenset({
    "kubeconform", "docker", "python", "sh", "shellcheck", "uv", "tar", "ls",
    "echo", "git", "mkdir", "cp", "sed", "grep", "bash",
})
# Snapshot @eb9ea18 (2026-08-25, review 2026-10-01): every first token
# observable in today's run: blocks beyond TOKEN_FLOOR — control-flow words,
# the gzip pipe, shell variable assignments, and the python -c string
# fragments of the config-load boot step (verbatim, however inelegant).
TOKEN_BASELINE = frozenset({
    "set", "for", "do", "done", "if", "fi", "gzip", "import", "from", "cfg",
    "hpo", "assert", "print('config", 'ref="${{', 'registry="ghcr.io/${{',
    (
        "Path('/app/project/"
        "config/experiments/mlflow.yaml').read_text())['hpo'])"
    ),
})


def collect_run_lines(workflow: dict[str, object]) -> list[str]:
    """Logical shell lines from every run: block, backslash-continuations joined."""
    logical: list[str] = []
    pending = ""
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "run" not in step:
                continue
            for raw in str(step["run"]).splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith("\\"):
                    pending += line[:-1] + " "
                    continue
                logical.append((pending + line).strip())
                pending = ""
    if pending:
        logical.append(pending.strip())
    return logical


def divergent_run_lines(
    lines: Sequence[str], allowed_tokens: frozenset[str], baseline_tokens: frozenset[str],
) -> list[str]:
    """Run-lines whose command head escapes floor ∪ baseline ∪ SSOT-invocation."""
    offenders = []
    for line in lines:
        if line.startswith(RUN_LOCAL_CI_INVOCATION):
            continue
        head = line.split()[0]
        if head not in allowed_tokens and head not in baseline_tokens:
            offenders.append(line)
    return offenders


def assert_gate_divergence(offenders: Sequence[str]) -> None:
    """RED with remediation guidance when CI grows an unbasetlined command."""
    if offenders:
        raise AssertionError(
            f"{CI_WORKFLOW}: run-line(s) outside the dated allowlist∪baseline: "
            f"{list(offenders)} — baseline the new command head(s) in TOKEN_BASELINE "
            "(or extend TOKEN_FLOOR by ruling) with a fresh review date"
        )


def test_probe_j_gate_divergence_watcher_live_and_falsifiable() -> None:
    """LIVE: zero divergent lines today; novel command heads go RED."""
    workflow = yaml.safe_load((ROOT / CI_WORKFLOW).read_text(encoding="utf-8"))
    lines = collect_run_lines(workflow)
    offenders = divergent_run_lines(lines, TOKEN_FLOOR, TOKEN_BASELINE)
    print(f"[P5] ci.yml run-lines={len(lines)} divergent={offenders}")
    assert_gate_divergence(offenders)
    # Negative controls (synthetic): a foreign package manager and a novel
    # runner head must both classify divergent...
    assert divergent_run_lines(["npm ci", "uvx ruff check"], TOKEN_FLOOR, TOKEN_BASELINE) == [
        "npm ci", "uvx ruff check",
    ]
    with pytest.raises(AssertionError, match="outside the dated allowlist"):
        assert_gate_divergence(["pip install requests"])
    # ...while the SSOT gate invocation passes WITHOUT being in either set...
    assert divergent_run_lines(
        ["bash scripts/run_local_ci.sh", "bash scripts/run_local_ci.sh --tier=fast"],
        TOKEN_FLOOR, TOKEN_BASELINE,
    ) == []
    # ...and a baselined exotic head stays green.
    assert divergent_run_lines(['ref="${{ github.ref }}"'], TOKEN_FLOOR, TOKEN_BASELINE) == []
