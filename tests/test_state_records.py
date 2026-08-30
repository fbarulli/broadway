"""Private STATE mirror regressions."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "state_records_under_test", REPO / "agents/tools/state_records.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state() -> str:
    return """# STATE.md

## CURRENT

| id | kind | status | owner | updated | source | github_item | mirror_state | summary |
|---|---|---|---|---|---|---|---|---|
| STATE-1 | checkpoint | open | main | 2026-08-30 | test | pending | pending | mirror me |

## Access protocol

private

## EVENTS

immutable-event-bytes
"""


def test_sync_paginates_skips_non_draft_and_preserves_events(tmp_path, monkeypatch) -> None:
    tool = _tool()
    state = tmp_path / "STATE.md"
    state.write_text(_state(), encoding="utf-8")
    before_events = state.read_text(encoding="utf-8").split("## EVENTS\n", 1)[1]
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *after=next*) printf '%s\\n' '{\"data\":{\"node\":{\"items\":{\"nodes\":[{\"id\":\"item-1\",\"content\":{\"__typename\":\"DraftIssue\",\"id\":\"draft-1\",\"body\":\"STATE: STATE-1\\nold\"}}],\"pageInfo\":{\"hasNextPage\":false,\"endCursor\":null}}}}}' ;;\n"
        "  *'mutation($draft'*) printf '%s\\n' '{\"data\":{\"updateProjectV2DraftIssue\":{\"draftIssue\":{\"id\":\"draft-1\"}}}}' ;;\n"
        "  *) case \"$*\" in *'nodes{id content{__typename ... on DraftIssue{id title body}}} pageInfo{hasNextPage endCursor}'*) ;; *) exit 9;; esac; printf '%s\\n' '{\"data\":{\"node\":{\"items\":{\"nodes\":[{\"id\":\"empty\",\"content\":null},{\"id\":\"ignored\",\"content\":{\"__typename\":\"Issue\",\"id\":\"issue\"}}],\"pageInfo\":{\"hasNextPage\":true,\"endCursor\":\"next\"}}}}}' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setattr(tool, "STATE", state)
    monkeypatch.setattr(sys, "argv", ["state_records.py", "record", "sync", "STATE-1"])
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    tool.main()
    text = state.read_text(encoding="utf-8")
    assert "| STATE-1 | checkpoint | open | main |" in text
    assert "| item-1 | synced |" in text
    assert text.split("## EVENTS\n", 1)[1] == before_events


def test_current_writes_reject_table_breaking_fields(tmp_path) -> None:
    tool = _tool()
    state = tmp_path / "STATE.md"
    state.write_text(_state(), encoding="utf-8")
    tool.STATE = state
    before, records, after = tool._load()
    records[0].values["summary"] = "bad | value"
    with pytest.raises(ValueError, match="pipes or newlines"):
        tool._write(before, records, after)


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
