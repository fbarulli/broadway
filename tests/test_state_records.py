"""State-records tool regressions — the rewrite's invariants, pinned.

Covers the transactional lifecycle: add/update/close/void, invalid
transitions, revision conflict, immutable ## EVENTS, journal recovery,
archive collision/idempotency, deterministic rendering, JSON output, and
terminal archive behavior. The tool under test is
agents/tools/state_records.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

from broadway.paths import repo_root

REPO = repo_root()
TOOL = REPO / "agents/tools/state_records.py"


def _load_tool() -> object:
    spec = importlib.util.spec_from_file_location("state_records_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_state(rows: str = "") -> str:
    """A minimal STATE.md CURRENT section (branch-tracked schema)."""
    header = "| id | kind | status | owner | custody | updated | source | summary |"
    divider = "|---|---|---|---|---|---|---|---|"
    row = ("| STATE-1 | checkpoint | open | main agent | main agent | 2026-08-30 | "
           "test | coordinate me |")
    return (
        "# STATE.md — current operational control record\n\n"
        "## CURRENT\n\n" + header + "\n" + divider + "\n" + row + rows + "\n"
        "## Access protocol\n\nprivate\n\n## Retention\n\nretain\n\n"
        "## EVENTS\n\nimmutable-event-bytes\n"
    )


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Wire the tool to a temporary branch-like STATE.md."""
    tool = _load_tool()
    state = tmp_path / "STATE.md"
    state.write_text(_make_state(), encoding="utf-8")
    monkeypatch.setattr(tool, "STATE_PATH", state)
    monkeypatch.setattr(tool, "ARCHIVE_DIR", tmp_path / "archive")
    return tool, None, state


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_add_persists_branch_tracked_row(env) -> None:
    tool, _, state = env
    tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                       status="open", owner="main agent", custody="main agent",
                                       source="test", summary="a decision"), "add")
    text = state.read_text(encoding="utf-8")
    assert "| STATE-20260901-001 | decision | open | main agent | main agent |" in text
    assert "github_item" not in text
    assert "mirror_state" not in text


def test_update_replaces_summary(env) -> None:
    tool, _, state = env
    tool.apply_record_operation(_args(env, "update", id="STATE-1", summary="new summary"), "update")
    text = state.read_text(encoding="utf-8")
    assert "new summary" in text


def test_close_marks_terminal_and_archives(env) -> None:
    tool, _, state = env
    tool.apply_record_operation(_args(env, "close", id="STATE-1", reason="done"), "close")
    text = state.read_text(encoding="utf-8")
    assert "| STATE-1" not in text  # terminal record removed from CURRENT
    # The reason lives in the archive, not in CURRENT.
    archived = "\n".join(p.read_text(encoding="utf-8") for p in tool.ARCHIVE_DIR.glob("*.md"))
    assert "STATE-ARCHIVE:STATE-1" in archived
    assert '"disposition": "CLOSED"' in archived
    assert '"reason": "done"' in archived


def test_void_marks_terminal_and_archives(env) -> None:
    tool, _, state = env
    tool.apply_record_operation(_args(env, "void", id="STATE-1", reason="invalid"), "void")
    text = state.read_text(encoding="utf-8")
    assert "| STATE-1" not in text
    archived = "\n".join(p.read_text(encoding="utf-8") for p in tool.ARCHIVE_DIR.glob("*.md"))
    assert '"disposition": "VOID"' in archived
    assert '"reason": "invalid"' in archived


def test_duplicate_add_rejected_with_valid_format_id(env) -> None:
    tool, _, _ = env
    # Seed a second row with a valid STATE-YYYYMMDD-NNN id, then re-add it.
    tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                      status="open", owner="main agent", custody="main agent",
                                      source="test", summary="first"), "add")
    with pytest.raises(ValueError, match="duplicate STATE record"):
        tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                          status="open", owner="main agent", custody="main agent",
                                          source="test", summary="dup"), "add")


def test_unknown_id_rejected(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="unknown STATE record"):
        tool.apply_record_operation(_args(env, "update", id="STATE-999", summary="x"), "update")


# --------------------------------------------------------------------------- #
# Invalid transitions + validation
# --------------------------------------------------------------------------- #

def test_add_rejects_terminal_status(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="create open records"):
        tool.apply_record_operation(_args(env, "add", id="STATE-20260901-002", kind="decision",
                                          status="closed", owner="main agent", custody="main agent",
                                          source="test", summary="bad"), "add")


def test_update_rejects_terminal_status(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="reserved for record close or record void"):
        tool.apply_record_operation(_args(env, "update", id="STATE-1", status="void"), "update")


def test_bad_kind_rejected(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="kind must be"):
        tool.apply_record_operation(_args(env, "add", id="STATE-20260901-003", kind="nope",
                                          status="open", owner="main agent", custody="main agent",
                                          source="test", summary="bad"), "add")


def test_bad_status_rejected(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="invalid STATE status"):
        tool.apply_record_operation(_args(env, "add", id="STATE-20260901-004", kind="decision",
                                          status="weird", owner="main agent", custody="main agent",
                                          source="test", summary="bad"), "add")


def test_pipe_in_fields_rejected(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="pipes or newlines"):
        tool.apply_record_operation(_args(env, "update", id="STATE-1", summary="bad | pipe"), "update")


def test_terminal_reason_forbids_pipes_and_semicolons(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="cannot contain"):
        tool.apply_record_operation(_args(env, "close", id="STATE-1", reason="a;b"), "close")


# --------------------------------------------------------------------------- #
# Revision conflict + immutable EVENTS
# --------------------------------------------------------------------------- #

def test_revision_conflict_blocks_stale_write(env) -> None:
    tool, _, state = env
    snapshot = tool._load_snapshot()
    # Mutate the file behind the snapshot's back.
    state.write_text(state.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="revision conflict"):
        tool._commit_current(snapshot, snapshot.records)


def test_events_mutation_refused(env) -> None:
    tool, _, state = env
    # Tamper EVENTS, then attempt to commit a FRESH snapshot of the tampered
    # file: _commit_current must refuse because the on-disk ## EVENTS no
    # longer matches the snapshot's events bytes.
    text = state.read_text(encoding="utf-8")
    state.write_text(text.replace("immutable-event-bytes", "TAMPERED"), encoding="utf-8")
    tampered = tool._load_snapshot()  # fresh snapshot now sees TAMPERED
    tampered = tool.StateSnapshot(tampered.prefix, tampered.records, tampered.suffix, "immutable-event-bytes", tampered.revision)
    with pytest.raises(RuntimeError, match="## EVENTS"):
        tool._commit_current(tampered, tampered.records)


# --------------------------------------------------------------------------- #
# Journal recovery
# --------------------------------------------------------------------------- #

def test_journal_recovered_on_next_lock(env) -> None:
    tool, _, _ = env
    # Simulate a crash mid-terminal: journal written, archive not applied.
    record = tool._load_snapshot().records[0]
    terminal = tool.TerminalRequest("CLOSED", "crash")
    archive = tool._archive_path(record)
    tool._write_journal(record, terminal, archive)
    assert tool._control_path("journal").exists()
    # Next locked operation runs recovery.
    with tool._state_lock():
        pass
    assert not tool._control_path("journal").exists()
    assert record.id in tool._archive_ids()


# --------------------------------------------------------------------------- #
# Archive collision / idempotency
# --------------------------------------------------------------------------- #

def test_archive_append_is_idempotent(env) -> None:
    tool, _, _ = env
    record = tool._load_snapshot().records[0]
    terminal = tool.TerminalRequest("CLOSED", "twice")
    path = tool._archive_path(record)
    tool._append_archive_once(path, record, terminal)
    tool._append_archive_once(path, record, terminal)  # same bytes -> no-op
    assert path.read_text(encoding="utf-8").count("STATE-ARCHIVE:" + record.id) == 1


def test_archive_identity_collision_rejected(env) -> None:
    tool, _, _ = env
    record = tool._load_snapshot().records[0]
    terminal_a = tool.TerminalRequest("CLOSED", "a")
    terminal_b = tool.TerminalRequest("CLOSED", "b")
    path = tool._archive_path(record)
    tool._append_archive_once(path, record, terminal_a)
    with pytest.raises(RuntimeError, match="identity collision"):
        tool._append_archive_once(path, record, terminal_b)


# --------------------------------------------------------------------------- #
# Deterministic rendering + JSON
# --------------------------------------------------------------------------- #

def test_render_is_deterministic(env) -> None:
    tool, _, _ = env
    a = tool._table(tool._load_snapshot().records)
    b = tool._table(tool._load_snapshot().records)
    assert a == b


def test_json_output(env) -> None:
    tool, _, _ = env
    record = tool._load_snapshot().records[0]
    tool._show(record, as_json=True)  # must not raise; JSON-serializable values
    assert json.loads(json.dumps(record.values, sort_keys=True))["id"] == "STATE-1"


# --------------------------------------------------------------------------- #
# Deterministic archive naming
# --------------------------------------------------------------------------- #

def test_archive_path_derives_month_from_record_updated(env) -> None:
    tool, _, _ = env
    record = tool._load_snapshot().records[0]  # updated=2026-08-30
    path = tool._archive_path(record)
    assert path.name == "2026-08.md"
    assert path.parent == tool.ARCHIVE_DIR


def test_archive_path_is_deterministic_across_calls(env) -> None:
    tool, _, _ = env
    record = tool._load_snapshot().records[0]
    assert tool._archive_path(record) == tool._archive_path(record)


# --------------------------------------------------------------------------- #
# Dry-run
# --------------------------------------------------------------------------- #

def test_dry_run_add_changes_nothing(env, capsys) -> None:
    tool, _, state = env
    before = state.read_text(encoding="utf-8")
    tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                      status="open", owner="main agent", custody="main agent",
                                      source="test", summary="planned", dry_run=True), "add")
    assert state.read_text(encoding="utf-8") == before  # file untouched
    out = capsys.readouterr().out
    assert "STATE DRY-RUN" in out
    assert '"id": "STATE-20260901-001"' in out


def test_dry_run_close_changes_nothing(env, capsys) -> None:
    tool, _, state = env
    before = state.read_text(encoding="utf-8")
    tool.apply_record_operation(_args(env, "close", id="STATE-1", reason="planned-close", dry_run=True), "close")
    assert state.read_text(encoding="utf-8") == before
    out = capsys.readouterr().out
    assert "terminal disposition CLOSED" in out


def test_dry_run_still_validates(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="unknown STATE record"):
        tool.apply_record_operation(_args(env, "close", id="STATE-999", reason="x", dry_run=True), "close")


# --------------------------------------------------------------------------- #
# Repo-layout guard (wrong-CWD protection)
# --------------------------------------------------------------------------- #

def test_layout_guard_rejects_missing_state(env, monkeypatch) -> None:
    tool, _, state = env
    state.unlink()
    with pytest.raises(RuntimeError, match="STATE_PATH missing"):
        tool._validate_repo_layout()


def test_mutation_output_exposes_state_path(env, capsys) -> None:
    tool, _, _ = env
    tool.apply_record_operation(_args(env, "update", id="STATE-1", summary="updated"), "update")
    out = capsys.readouterr().out
    assert "state_path:" in out
    assert "STATE.md" in out


def test_cli_has_no_network_mirror_commands() -> None:
    help_text = _load_tool().parser().format_help()
    assert "sync" not in help_text
    assert "reconcile" not in help_text


# --------------------------------------------------------------------------- #
# CURRENT-table law (schema-aware)
# --------------------------------------------------------------------------- #

def test_active_state_has_no_legacy_tail_before_events() -> None:
    active = (REPO / "agents/ledger/STATE.md").read_text(encoding="utf-8").split(
        "\n## EVENTS\n", 1,
    )[0]
    assert re.findall(r"^#{1,3} .+$", active, flags=re.MULTILINE) == [
        "# STATE.md — current operational control record",
        "## CURRENT",
        "## Access protocol",
        "## Retention",
    ]
    table_lines = [line for line in active.splitlines() if line.startswith("|")]
    assert table_lines[0] == "| id | kind | status | owner | custody | updated | source | summary |"
    assert all(len(line.strip("|").split("|")) == 8 for line in table_lines)
    # Frozen 2026-09-22: CURRENT carries no open rows (closed lanes live in
    # agents/ledger/archive/). Only the header plus divider remain.
    assert len(table_lines) == 2
    assert "github_item" not in active
    assert "mirror_state" not in active


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _args(env, operation: str, **fields) -> object:
    """Build a namespace with the fields a CLI invocation would supply."""
    class _A:
        pass
    a = _A()
    for key, value in fields.items():
        setattr(a, key, value)
    a.json = False
    a.reason = fields.get("reason", "")
    a.dry_run = fields.get("dry_run", False)
    return a
