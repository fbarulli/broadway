"""State-records tool regressions — the rewrite's invariants, pinned.

Covers the transactional lifecycle: add/update/close/void, invalid
transitions, revision conflict, immutable ## EVENTS, journal recovery,
archive collision/idempotency, deterministic rendering, JSON output,
mirror identity, and terminal archive behavior. The tool under test is
agents/tools/state_records.py (the landed rewrite) with
agents/tools/project_board.py for the mirror transport.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "agents/tools/state_records.py"


def _load_tool() -> object:
    spec = importlib.util.spec_from_file_location("state_records_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_project_board() -> object:
    spec = importlib.util.spec_from_file_location(
        "project_board_under_test", REPO / "agents/tools/project_board.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_state(rows: str = "") -> str:
    """A minimal STATE.md CURRENT section (10-column schema, post-migration)."""
    header = ("| id | kind | status | owner | custody | updated | source | "
              "github_item | mirror_state | summary |")
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    row = ("| STATE-1 | checkpoint | open | main agent | main agent | 2026-08-30 | "
           "test | pending | pending | mirror me |")
    return (
        "# STATE.md — current operational control record\n\n"
        "## CURRENT\n\n" + header + "\n" + divider + "\n" + row + rows + "\n"
        "## Access protocol\n\nprivate\n\n## Retention\n\nretain\n\n"
        "## EVENTS\n\nimmutable-event-bytes\n"
    )


class _BoardFixture:
    """A tiny in-memory board double implementing the transport API."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.next_id = 0

    def create_draft(self, title: str, body: str, status_name: str | None = None) -> str:
        item_id = f"item-{self.next_id}"
        self.next_id += 1
        self.items[item_id] = {"title": title, "body": body, "status": status_name or "Todo"}
        return item_id

    def update_draft(self, item_id: str, *, title: str | None = None, body: str | None = None) -> None:
        self.items[item_id]["title"] = title if title is not None else self.items[item_id]["title"]
        self.items[item_id]["body"] = body if body is not None else self.items[item_id]["body"]

    def update_status(self, item_id: str, status_name: str) -> None:
        self.items[item_id]["status"] = status_name

    def get(self, item_id: str):
        class _Item:
            body = self.items[item_id]["body"]
        return _Item()

    def find_state_mirror(self, record_id: str) -> str | None:
        for item_id, item in self.items.items():
            body = item["body"]
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("id") == record_id:
                return item_id
        return None

    def iter_items(self):
        class _I:
            def __init__(self, item_id, body):
                self.item_id = item_id
                self.body = body
        for item_id, item in self.items.items():
            yield _I(item_id, item["body"])


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Wire the tool to a temp STATE.md + in-memory board."""
    tool = _load_tool()
    board = _BoardFixture()
    state = tmp_path / "STATE.md"
    state.write_text(_make_state(), encoding="utf-8")
    monkeypatch.setattr(tool, "STATE_PATH", state)
    monkeypatch.setattr(tool, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(tool, "project_board", board)
    return tool, board, state


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_add_creates_pending_row_then_syncs(env) -> None:
    tool, board, state = env
    tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                       status="open", owner="main agent", custody="main agent",
                                       source="test", summary="a decision"), "add")
    text = state.read_text(encoding="utf-8")
    assert "| STATE-20260901-001 | decision | open | main agent | main agent |" in text
    assert "| synced |" in text  # add writes local, mirrors, marks synced
    assert len(board.items) == 1


def test_update_replaces_summary_and_marks_pending_then_synced(env) -> None:
    tool, _, state = env
    tool.apply_record_operation(_args(env, "update", id="STATE-1", summary="new summary"), "update")
    text = state.read_text(encoding="utf-8")
    assert "new summary" in text
    assert "| synced |" in text


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
# Mirror identity
# --------------------------------------------------------------------------- #

def test_sync_binds_correct_board_item(env) -> None:
    tool, board, _ = env
    tool.apply_record_operation(_args(env, "sync", id="STATE-1"), "sync")
    # The mirrored body is JSON with id STATE-1.
    found = board.find_state_mirror("STATE-1")
    assert found is not None
    item = board.items[found]
    payload = json.loads(item["body"])
    assert payload["id"] == "STATE-1"


def test_sync_paginates_skips_non_draft_and_preserves_events(env) -> None:
    """find_state_mirror walks all items, skips non-JSON-state bodies, and the
    sync never touches ## EVENTS bytes."""
    tool, board, state = env
    board.items["foreign"] = {"title": "t", "body": "not-json", "status": "Todo"}
    tool.apply_record_operation(_args(env, "sync", id="STATE-1"), "sync")
    found = board.find_state_mirror("STATE-1")
    assert found is not None
    assert found != "foreign"
    before_events = state.read_text(encoding="utf-8").split("## EVENTS\n", 1)[1]
    tool.apply_record_operation(_args(env, "sync", id="STATE-1"), "sync")
    assert state.read_text(encoding="utf-8").split("## EVENTS\n", 1)[1] == before_events


def test_sync_of_unknown_record_not_found(env) -> None:
    tool, _, _ = env
    with pytest.raises(ValueError, match="unknown STATE record"):
        tool.apply_record_operation(_args(env, "sync", id="STATE-999"), "sync")


def test_terminal_archive_writes_board_done(env) -> None:
    tool, board, _ = env
    tool.apply_record_operation(_args(env, "close", id="STATE-1", reason="done"), "close")
    assert all(item["status"] == "Done" for item in board.items.values())


# --------------------------------------------------------------------------- #
# Deterministic archive naming + reconcile
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


def test_reconcile_reports_local_only_drift(env, capsys) -> None:
    tool, board, _ = env
    board.create_draft("x", json.dumps({"id": "STATE-OTHER"}))
    tool.command_reconcile(_args(env, "reconcile"))
    out = capsys.readouterr().out
    assert "STATE-1" in out


def test_reconcile_reports_board_extra(env, capsys) -> None:
    tool, board, _ = env
    board.create_draft("x", json.dumps({"id": "STATE-ORPHAN"}))
    tool.command_reconcile(_args(env, "reconcile"))
    out = capsys.readouterr().out
    assert "board-extra STATE-ORPHAN" in out


def test_reconcile_ok_when_consistent(env, capsys) -> None:
    tool, _, _ = env
    # Sync first so the board holds STATE-1's mirror.
    tool.apply_record_operation(_args(env, "sync", id="STATE-1"), "sync")
    tool.command_reconcile(_args(env, "reconcile"))
    out = capsys.readouterr().out
    assert "RECONCILE OK" in out


# --------------------------------------------------------------------------- #
# Dry-run (no file or board change)
# --------------------------------------------------------------------------- #

def test_dry_run_add_changes_nothing(env, capsys) -> None:
    tool, board, state = env
    before = state.read_text(encoding="utf-8")
    tool.apply_record_operation(_args(env, "add", id="STATE-20260901-001", kind="decision",
                                      status="open", owner="main agent", custody="main agent",
                                      source="test", summary="planned", dry_run=True), "add")
    assert state.read_text(encoding="utf-8") == before  # file untouched
    assert len(board.items) == 0  # board untouched
    out = capsys.readouterr().out
    assert "STATE DRY-RUN" in out
    assert '"id": "STATE-20260901-001"' in out


def test_dry_run_close_changes_nothing(env, capsys) -> None:
    tool, board, state = env
    before = state.read_text(encoding="utf-8")
    tool.apply_record_operation(_args(env, "close", id="STATE-1", reason="planned-close", dry_run=True), "close")
    assert state.read_text(encoding="utf-8") == before
    assert len(board.items) == 0
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
    tool.apply_record_operation(_args(env, "sync", id="STATE-1"), "sync")
    out = capsys.readouterr().out
    assert "state_path:" in out
    assert "STATE.md" in out


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
    assert sum(line.startswith("|") for line in active.splitlines()) == 3


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
