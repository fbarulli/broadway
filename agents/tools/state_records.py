"""Controlled STATE lifecycle and GitHub Project mirror."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
try:
    import project_board  # sibling module, resolved relative to THIS file
except ModuleNotFoundError:  # pragma: no cover - fallback for unusual layouts
    from agents.tools import project_board

REPO = Path(__file__).resolve().parents[2]
STATE_PATH = REPO / "agents" / "ledger" / "STATE.md"
ARCHIVE_DIR = REPO / "agents" / "ledger" / "archive"
CURRENT_START, CURRENT_END, EVENTS_START = "## CURRENT\n", "## Access protocol\n", "## EVENTS\n"
COLUMNS = ("id", "kind", "status", "owner", "custody", "updated", "source", "github_item", "mirror_state", "summary")
KINDS = {"lane", "custody", "hazard", "decision", "checkpoint"}
STATE_ID = re.compile(r"STATE-\d{8}-\d{3}")
STATUSES = frozenset({"open", "blocked", "approved", "closed", "void"})


@dataclass(frozen=True)
class Record:
    values: dict[str, str]
    @property
    def id(self) -> str: return self.values["id"]


@dataclass(frozen=True)
class StateSnapshot:
    prefix: str
    records: tuple[Record, ...]
    suffix: str
    events: str
    revision: str


@dataclass(frozen=True)
class TerminalRequest:
    disposition: Literal["CLOSED", "VOID"]
    reason: str

    def __post_init__(self) -> None:
        if self.disposition not in {"CLOSED", "VOID"}:
            raise ValueError("terminal disposition must be CLOSED or VOID")
        if not self.reason.strip() or any(token in self.reason for token in ("|", "\n", "\r", ";")):
            raise ValueError("terminal reason must be nonempty and cannot contain pipes, newlines, or semicolons")


def _digest(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()


def _control_path(kind: str) -> Path:
    return Path(tempfile.gettempdir()) / f"broadway-state-{_digest(str(STATE_PATH.resolve()))[:16]}.{kind}"


def _status(value: str) -> str:
    normalized = value.strip()
    if normalized not in STATUSES:
        raise ValueError(f"invalid STATE status: {value!r}")
    return normalized


def _split_document(text: str) -> tuple[str, str, str, str]:
    start, end, events = text.find(CURRENT_START), text.find(CURRENT_END), text.find(EVENTS_START)
    if min(start, end, events) < 0 or start >= end < events:
        raise ValueError("STATE.md must contain ordered ## CURRENT, ## Access protocol, and ## EVENTS sections")
    return text[:start], text[start:end], text[end:events], text[events:]


def _parse_table(current: str) -> list[Record]:
    records = []
    for line in [line for line in current.splitlines() if line.startswith("|")][2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS) or any("\n" in cell or "|" in cell for cell in cells):
            raise ValueError(f"malformed CURRENT record: {line}")
        values = dict(zip(COLUMNS, cells, strict=True)); values["status"] = _status(values["status"])
        records.append(Record(values))
    if len({record.id for record in records}) != len(records): raise ValueError("CURRENT record ids must be unique")
    return records


def _table(records: Sequence[Record]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |\n"
    divider = "|" + "|".join("---" for _ in COLUMNS) + "|\n"
    rows = ["| " + " | ".join(record.values[column] for column in COLUMNS) + " |\n" for record in records]
    return CURRENT_START + "\n" + header + divider + "".join(rows) + "\n"


def _load_snapshot() -> StateSnapshot:
    _validate_repo_layout()
    text = STATE_PATH.read_text(encoding="utf-8")
    prefix, current, suffix, events = _split_document(text)
    return StateSnapshot(prefix, tuple(_parse_table(current)), suffix, events, _digest(text))


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); handle.write(text); handle.flush(); os.fsync(handle.fileno())
    try:
        temporary.replace(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: temporary.unlink(missing_ok=True)


def _commit_current(snapshot: StateSnapshot, records: Sequence[Record]) -> StateSnapshot:
    text = STATE_PATH.read_text(encoding="utf-8")
    if _digest(text) != snapshot.revision: raise RuntimeError("STATE revision conflict; re-run from a fresh snapshot")
    if _split_document(text)[3] != snapshot.events: raise RuntimeError("refusing STATE mutation: ## EVENTS changed during operation")
    _atomic_write(STATE_PATH, snapshot.prefix + _table(records) + snapshot.suffix + snapshot.events)
    committed = _load_snapshot()
    if committed.events != snapshot.events: raise RuntimeError("STATE write violated immutable ## EVENTS boundary")
    return committed


def _find(records: Sequence[Record], record_id: str) -> tuple[int, Record]:
    for index, record in enumerate(records):
        if record.id == record_id: return index, record
    raise ValueError(f"unknown STATE record: {record_id}")


def _replace(records: Sequence[Record], index: int, record: Record) -> list[Record]:
    result = list(records); result[index] = record; return result


def _now() -> str: return datetime.now(UTC).date().isoformat()
def _no_item(value: str) -> bool: return value in {"", "pending"}


def _mirror_title(record: Record) -> str: return f"[{record.values['status'].upper()}] {record.id} {record.values['kind']}"


def _sync_mirror(record: Record) -> str:
    item_id = record.values["github_item"]
    if not _no_item(item_id):
        try:
            body = json.loads(project_board.get(item_id).body)
            if not isinstance(body, dict) or body.get("id") != record.id:
                raise ValueError("stored item is not this mirror")
        except (project_board.ProjectBoardError, ValueError, json.JSONDecodeError):
            item_id = project_board.find_state_mirror(record.id)
            if item_id is None:
                raise RuntimeError(f"stored GitHub item cannot be verified for {record.id}")
    else:
        item_id = project_board.find_state_mirror(record.id)
    if item_id is None or _no_item(item_id):
        item_id = project_board.create_draft(_mirror_title(record), json.dumps(record.values, sort_keys=True))
    values = dict(record.values); values.update(github_item=item_id, mirror_state="synced"); final = Record(values)
    project_board.update_draft(item_id, title=_mirror_title(final), body=json.dumps(final.values, sort_keys=True))
    if final.values["status"] in {"closed", "void"}:
        project_board.update_status(item_id, "Done")
    return item_id


def _archive_path(record: Record) -> Path:
    """Deterministic archive file: month derives from the record's OWN
    persisted `updated` date, never wall-clock now — two terminal records in
    one run cannot collide on a month boundary, and rendering is stable."""
    updated = record.values.get("updated", _now())
    month = updated[:7]  # YYYY-MM from the record's own date
    return ARCHIVE_DIR / f"{month}.md"


def _archive_entry(record: Record, terminal: TerminalRequest) -> str:
    payload = {"record": record.values, "terminal": {"disposition": terminal.disposition, "reason": terminal.reason}}
    return f"<!-- STATE-ARCHIVE:{record.id} -->\n{json.dumps(payload, sort_keys=True)}\n"


def _archive_ids() -> set[str]:
    found: set[str] = set()
    for path in ARCHIVE_DIR.glob("*.md"):
        ids = re.findall(r'"id":\s*"(STATE-[^"]+)"|STATE-ARCHIVE:(STATE-[^>]+)', path.read_text(encoding="utf-8"))
        for record_id in {json_id or marker_id for json_id, marker_id in ids}:
            if record_id in found: raise ValueError(f"duplicate archived STATE id: {record_id}")
            found.add(record_id)
    return found


def _append_archive_once(path: Path, record: Record, terminal: TerminalRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# State archive\n\n**Class:** SSOT\n\n"
    marker = f"<!-- STATE-ARCHIVE:{record.id} -->"
    if marker in existing:
        expected = _archive_entry(record, terminal)
        if existing.count(expected) != 1 or existing.count(marker) != 1: raise RuntimeError(f"archive identity collision: {record.id}")
        return
    if record.id in _archive_ids():
        raise RuntimeError(f"archive identity collision: {record.id}")
    _atomic_write(path, existing + _archive_entry(record, terminal))


def _write_journal(record: Record, terminal: TerminalRequest, archive: Path) -> None:
    _atomic_write(_control_path("journal"), json.dumps({"record": record.values, "disposition": terminal.disposition, "reason": terminal.reason, "archive": str(archive)}, sort_keys=True))


def _recover_journal() -> None:
    journal = _control_path("journal")
    if not journal.exists(): return
    payload = json.loads(journal.read_text(encoding="utf-8")); record = Record(payload["record"])
    terminal = TerminalRequest(payload["disposition"], payload["reason"]); _append_archive_once(Path(payload["archive"]), record, terminal)
    snapshot = _load_snapshot()
    try: index, current = _find(snapshot.records, record.id)
    except ValueError: pass
    else:
        if current != record: raise RuntimeError(f"journal identity conflict for {record.id}")
        _commit_current(snapshot, [item for position, item in enumerate(snapshot.records) if position != index])
    journal.unlink()


def _validate_repo_layout() -> None:
    """Fail loudly if the resolved repo layout is not the expected one.

    STATE_PATH is derived from the TOOL's own location (never cwd), so a
    wrong-cwd invocation cannot silently mutate a stranger's STATE.md. This
    guard additionally rejects a checkout whose markers are missing.
    """
    if not (REPO / ".git").exists() and not (REPO / ".git").is_dir() and not (REPO / ".git").is_file():
        raise RuntimeError(f"STATE ERROR: {REPO} is not a git worktree (no .git)")
    if not STATE_PATH.exists():
        raise RuntimeError(f"STATE ERROR: resolved STATE_PATH missing: {STATE_PATH}")
    if not (REPO / "agents" / "contracts").is_dir():
        raise RuntimeError(f"STATE ERROR: {REPO}/agents/contracts missing — wrong checkout?")


@contextmanager
def _state_lock() -> Iterator[None]:
    lock = _control_path("lock"); lock.touch(exist_ok=True)
    with lock.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try: _recover_journal(); yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def _archive_terminal(snapshot: StateSnapshot, record: Record, terminal: TerminalRequest) -> None:
    archive = _archive_path(record); _write_journal(record, terminal, archive); _append_archive_once(archive, record, terminal)
    index, _ = _find(snapshot.records, record.id)
    _commit_current(snapshot, [item for position, item in enumerate(snapshot.records) if position != index]); _control_path("journal").unlink()


def _record_from_args(args: argparse.Namespace) -> Record:
    values = {column: getattr(args, column, "") or "" for column in COLUMNS}
    values.update(status=_status(values["status"]), updated=_now(), github_item="pending", mirror_state="pending")
    if not STATE_ID.fullmatch(values["id"]) or values["kind"] not in KINDS: raise ValueError("id must match STATE-YYYYMMDD-NNN and kind must be lane/custody/hazard/decision/checkpoint")
    if values["status"] in {"close", "closed", "void"}: raise ValueError("create open records, then use record close or record void")
    if any("|" in value or "\n" in value for value in values.values()): raise ValueError("CURRENT values cannot contain pipes or newlines")
    return Record(values)


def _show(record: Record, as_json: bool) -> None:
    if as_json:
        payload = dict(record.values); payload["state_path"] = str(STATE_PATH)
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"state_path: {STATE_PATH}")
        print("\n".join(f"{key}: {value}" for key, value in record.values.items()))


def _terminal_from_record(record: Record) -> TerminalRequest | None:
    status = record.values["status"]
    if status not in {"closed", "void"}: return None
    prefix = f"{status.upper()}: "
    summary = record.values["summary"]
    if not summary.startswith(prefix): raise ValueError(f"terminal STATE record {record.id} lacks a recoverable reason")
    disposition: Literal["CLOSED", "VOID"] = "CLOSED" if status == "closed" else "VOID"
    return TerminalRequest(disposition, summary[len(prefix):].partition(";")[0])


def apply_record_operation(args: argparse.Namespace, operation: Literal["add", "update", "close", "void", "sync"]) -> None:
    """Only STATE mutation boundary: local intent, mirror, then terminal archive."""
    _validate_repo_layout()
    with _state_lock():
        snapshot = _load_snapshot(); terminal: TerminalRequest | None = None
        planned: list[Record]
        if operation == "add":
            record = _record_from_args(args)
            if any(item.id == record.id for item in snapshot.records) or record.id in _archive_ids(): raise ValueError(f"duplicate STATE record: {record.id}")
            planned = [*snapshot.records, record]
        else:
            index, current = _find(snapshot.records, args.id); values = dict(current.values)
            if operation == "update":
                if getattr(args, "status", None) in {"close", "closed", "void"}:
                    raise ValueError("terminal status is reserved for record close or record void")
                for field in ("status", "owner", "custody", "source", "summary"):
                    value = getattr(args, field, None)
                    if value is not None:
                        if "|" in value or "\n" in value: raise ValueError(f"{field} cannot contain pipes or newlines")
                        values[field] = _status(value) if field == "status" else value
                values.update(updated=_now(), mirror_state="pending")
                record = Record(values); planned = _replace(snapshot.records, index, record)
            elif operation in {"close", "void"}:
                terminal = TerminalRequest("CLOSED" if operation == "close" else "VOID", args.reason)
                values.update(status=terminal.disposition.lower(), summary=f"{terminal.disposition}: {args.reason}; {values['summary']}", updated=_now(), mirror_state="pending")
                record = Record(values); planned = _replace(snapshot.records, index, record)
            else:  # sync — no local delta until the mirror settles
                record = current; planned = list(snapshot.records)
        if getattr(args, "dry_run", False):
            print("STATE DRY-RUN: no file or board change was made; planned row:")
            print(json.dumps(record.values, sort_keys=True))
            if terminal:
                print(f"STATE DRY-RUN: terminal disposition {terminal.disposition} ({terminal.reason}) -> archive")
            return
        snapshot = _commit_current(snapshot, planned)
        if operation == "sync": terminal = _terminal_from_record(record)
        item_id = _sync_mirror(record)
        index, current = _find(snapshot.records, record.id); values = dict(current.values)
        values.update(github_item=item_id, mirror_state="synced", updated=_now()); record = Record(values)
        snapshot = _commit_current(snapshot, _replace(snapshot.records, index, record))
        if terminal: _archive_terminal(snapshot, record, terminal)
        _show(record, getattr(args, "json", False))


def command_get(args: argparse.Namespace) -> None: _show(_find(_load_snapshot().records, args.id)[1], args.json)
def command_list(args: argparse.Namespace) -> None:
    records = [record for record in _load_snapshot().records if (not args.kind or record.values["kind"] == args.kind) and (not args.status or record.values["status"] == args.status)]
    print(json.dumps([record.values for record in records], sort_keys=True) if args.json else _table(records), end="")
def command_snapshot(_: argparse.Namespace) -> None:
    snapshot = _load_snapshot(); print(_table(snapshot.records) + snapshot.suffix + f"EVENTS-SHA256: {_digest(snapshot.events)}\nREVISION-SHA256: {snapshot.revision}")
def command_check(_: argparse.Namespace) -> None:
    snapshot = _load_snapshot()
    if not snapshot.records: raise ValueError("CURRENT must contain at least one record")
    overlap = {record.id for record in snapshot.records} & _archive_ids()
    if overlap: raise ValueError(f"CURRENT/archive STATE id overlap: {sorted(overlap)}")
    print(f"STATE OK: {len(snapshot.records)} CURRENT record(s); EVENTS-SHA256 {_digest(snapshot.events)}")


def command_reconcile(_: argparse.Namespace) -> None:
    """Report local<->board drift WITHOUT repairing it.

    Classes reported (exit 0 always; a drift report is not a failure):
      local-only   — CURRENT/archived row has no board item (mirror lost)
      board-extra  — board item with no local counterpart (orphan)
      identity     — local row points at a board item that is not its mirror
      stale        — local row is pending (mirror never settled)
    """
    snapshot = _load_snapshot()
    local_ids = {record.values["id"] for record in snapshot.records} | _archive_ids()
    board_items = {item.item_id: item.body for item in project_board.iter_items()}
    board_by_id: dict[str, str] = {}
    for item_id, body in board_items.items():
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            board_by_id.setdefault(payload["id"], item_id)
    drift: list[str] = []
    for record in snapshot.records:
        item_id = record.values["github_item"]
        if record.values["mirror_state"] == "pending":
            drift.append(f"stale      {record.values['id']}: mirror pending (run record sync)")
            continue
        if item_id in {"", "pending"}:
            drift.append(f"local-only {record.values['id']}: no board item id recorded")
        elif item_id not in board_items:
            drift.append(f"local-only {record.values['id']}: board item {item_id} not found")
        elif board_by_id.get(record.values["id"]) != item_id:
            drift.append(f"identity   {record.values['id']}: item {item_id} is not its mirror")
    for record_id, item_id in board_by_id.items():
        if record_id not in local_ids:
            drift.append(f"board-extra {record_id}: orphan item {item_id}")
    if drift:
        print("\n".join(drift))
    else:
        print(f"STATE RECONCILE OK: {len(snapshot.records)} CURRENT + {len(board_items)} board item(s) consistent")


def command_template(args: argparse.Namespace) -> None: print(json.dumps({column: (args.kind if column == "kind" else "<value>") for column in COLUMNS}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__); sub = root.add_subparsers(dest="command", required=True)
    for name, function in (("get", command_get), ("list", command_list)):
        command = sub.add_parser(name)
        if name == "get": command.add_argument("id")
        else: command.add_argument("--kind", choices=sorted(KINDS)); command.add_argument("--status")
        command.add_argument("--json", action="store_true"); command.set_defaults(func=function)
    for name, command_function in (("snapshot", command_snapshot), ("check", command_check), ("reconcile", command_reconcile)):
        command = sub.add_parser(name); command.set_defaults(func=command_function)
    template = sub.add_parser("template"); template.add_argument("--kind", choices=sorted(KINDS), required=True); template.set_defaults(func=command_template)
    record = sub.add_parser("record", help="the only STATE mutation namespace"); record_sub = record.add_subparsers(dest="record_command", required=True)
    add = record_sub.add_parser("add")
    for field in ("id", "kind", "status", "owner", "custody", "source", "summary"): add.add_argument(f"--{field}", required=True)
    add.add_argument("--json", action="store_true"); add.add_argument("--dry-run", action="store_true"); add.set_defaults(func=lambda args: apply_record_operation(args, "add"))
    update = record_sub.add_parser("update"); update.add_argument("id")
    for field in ("status", "owner", "custody", "source", "summary"): update.add_argument(f"--{field}")
    update.add_argument("--json", action="store_true"); update.add_argument("--dry-run", action="store_true"); update.set_defaults(func=lambda args: apply_record_operation(args, "update"))
    for name in ("close", "void"):
        command = record_sub.add_parser(name); command.add_argument("id"); command.add_argument("--reason", required=True)
        command.add_argument("--dry-run", action="store_true"); command.set_defaults(func=lambda args, value=name: apply_record_operation(args, value))
    sync = record_sub.add_parser("sync"); sync.add_argument("id"); sync.add_argument("--json", action="store_true"); sync.set_defaults(func=lambda args: apply_record_operation(args, "sync"))
    return root


def main() -> None:
    args = parser().parse_args()
    try: args.func(args)
    except (OSError, ValueError, RuntimeError) as error: raise SystemExit(f"STATE ERROR: {error}") from error


if __name__ == "__main__": main()
