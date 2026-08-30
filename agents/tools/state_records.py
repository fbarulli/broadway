"""Main-only CURRENT lifecycle and its private Project #4 mirror."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "agents/ledger/STATE.md"
PROJECT_ID = "PVT_kwHOAZFnCc4Bhhjq"
COLUMNS = ("id", "kind", "status", "owner", "updated", "source", "github_item", "mirror_state", "summary")


@dataclass(frozen=True)
class Record:
    values: dict[str, str]

    @property
    def id(self) -> str:
        return self.values["id"]


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _split(text: str) -> tuple[str, str, str]:
    active, events = text.split("\n## EVENTS\n", 1)
    start = text.index("## CURRENT\n")
    end = active.index("## Access protocol\n")
    return text[:start], text[start:end], active[end:] + "\n## EVENTS\n" + events


def _records(current: str) -> list[Record]:
    rows = [line for line in current.splitlines() if line.startswith("| STATE-")]
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            raise ValueError(f"STATE ERROR: malformed CURRENT row: {row}")
        parsed.append(Record(dict(zip(COLUMNS, cells, strict=True))))
    if not parsed or len({record.id for record in parsed}) != len(parsed):
        raise ValueError("STATE ERROR: CURRENT ids must be non-empty and unique")
    return parsed


def _render(records: list[Record]) -> str:
    header = "## CURRENT\n\n| " + " | ".join(COLUMNS) + " |\n|" + "|".join("---" for _ in COLUMNS) + "|\n"
    return header + "".join("| " + " | ".join(record.values[key] for key in COLUMNS) + " |\n" for record in records) + "\n"


def _load() -> tuple[str, list[Record], str]:
    before, current, after = _split(STATE.read_text(encoding="utf-8"))
    return before, _records(current), after


def _write(before: str, records: list[Record], after: str) -> None:
    for record in records:
        if any("|" in value or "\n" in value for value in record.values.values()):
            raise ValueError("STATE ERROR: CURRENT fields cannot contain pipes or newlines")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=STATE.parent, delete=False) as fh:
        fh.write(before + _render(records) + after)
        temporary = Path(fh.name)
    temporary.replace(STATE)


def _find(records: list[Record], record_id: str) -> tuple[int, Record]:
    for index, record in enumerate(records):
        if record.id == record_id:
            return index, record
    raise ValueError(f"STATE ERROR: unknown record: {record_id}")


def _gh(query: str, **variables: str | None) -> dict:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    command.extend(f"-f={key}={value or ''}" for key, value in variables.items())
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"STATE ERROR: GitHub Project command failed ({result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout)


ITEMS_QUERY = """query($project:ID!,$after:String){node(id:$project){... on ProjectV2{items(first:100,after:$after){nodes{id content{__typename ... on DraftIssue{id title body}}} pageInfo{hasNextPage endCursor}}}}}"""
CREATE_MUTATION = """mutation($project:ID!,$title:String!,$body:String!){addProjectV2DraftIssue(input:{projectId:$project,title:$title,body:$body}){projectItem{id}}}"""
UPDATE_MUTATION = """mutation($draft:ID!,$title:String!,$body:String!){updateProjectV2DraftIssue(input:{draftIssueId:$draft,title:$title,body:$body}){draftIssue{id}}}"""


def _body(record: Record) -> str:
    return "STATE: " + record.id + "\n\n" + record.values["summary"]


def _mirror(record: Record) -> str:
    after: str | None = None
    found: tuple[str, str] | None = None
    while True:
        payload = _gh(ITEMS_QUERY, project=PROJECT_ID, after=after)
        items = payload["data"]["node"]["items"]
        for item in items["nodes"]:
            content = item.get("content")
            if not content or content.get("__typename") != "DraftIssue":
                continue
            if content.get("body", "").startswith(f"STATE: {record.id}\n"):
                found = (item["id"], content["id"])
                break
        if found or not items["pageInfo"]["hasNextPage"]:
            break
        after = items["pageInfo"]["endCursor"]
    title = f"[{record.values['status'].upper()}] {record.id}"
    if found:
        _gh(UPDATE_MUTATION, draft=found[1], title=title, body=_body(record))
        return found[0]
    payload = _gh(CREATE_MUTATION, project=PROJECT_ID, title=title, body=_body(record))
    return payload["data"]["addProjectV2DraftIssue"]["projectItem"]["id"]


def sync(record_id: str) -> None:
    before, records, after = _load()
    index, record = _find(records, record_id)
    pending = replace(record, values=record.values | {"mirror_state": "pending", "updated": _today()})
    records[index] = pending
    _write(before, records, after)
    item = _mirror(pending)
    before, records, after = _load()
    index, current = _find(records, record_id)
    records[index] = replace(current, values=current.values | {"github_item": item, "mirror_state": "synced", "updated": _today()})
    _write(before, records, after)


def add(args: argparse.Namespace) -> None:
    before, records, after = _load()
    if any(record.id == args.id for record in records):
        raise ValueError(f"STATE ERROR: duplicate record: {args.id}")
    values = {key: getattr(args, key) for key in ("id", "kind", "status", "owner", "source", "summary")}
    records.append(Record(values | {"updated": _today(), "github_item": "pending", "mirror_state": "pending"}))
    _write(before, records, after)
    sync(args.id)


def update(args: argparse.Namespace) -> None:
    before, records, after = _load()
    index, record = _find(records, args.id)
    records[index] = replace(record, values=record.values | {"summary": args.summary, "updated": _today(), "mirror_state": "pending"})
    _write(before, records, after)
    sync(args.id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.set_defaults(func=lambda _: print(STATE.read_text(encoding="utf-8"), end=""))
    record = commands.add_parser("record").add_subparsers(dest="action", required=True)
    for name, function in (("add", add), ("update", update)):
        command = record.add_parser(name)
        command.add_argument("id")
        if name == "add":
            for field in ("kind", "status", "owner", "source", "summary"):
                command.add_argument(f"--{field}", required=True)
        else:
            command.add_argument("--summary", required=True)
        command.set_defaults(func=function)
    sync_command = record.add_parser("sync")
    sync_command.add_argument("id")
    sync_command.set_defaults(func=lambda args: sync(args.id))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
