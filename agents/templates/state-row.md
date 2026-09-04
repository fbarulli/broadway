# TEMPLATE — STATE row (open a lane / record a decision / file a hazard)
#
# Copy this shape, fill every field, then create the row via the ONLY sanctioned
# mutation path (never hand-edit STATE.md):
#
#   uv run python agents/tools/state_records.py record add --id <id> \
#     --kind <kind> --status open --owner "<who>" --custody "<who holds it>" \
#     --source "<provenance>" --summary "<text>"
#
# (No --updated flag: the tool stamps the date itself — passing --updated is
# a usage error. Subcommand help is the interface truth: record add --help.)
#
# REQUIREMENTS (enforced by state_records.py record add):
#   R1. id        STATE-YYYYMMDD-NNN, unique, never reused (rows live in
#                 agents/ledger/STATE.md CURRENT; terminal rows archive with
#                 STATE-ARCHIVE:<id> markers under agents/ledger/archive/)
#   R2. kind      exactly one of: lane|custody|decision|hazard|checkpoint
#   R3. status    "open" when creating (terminal statuses only via close/void)
#   R4. owner     the deciding authority: "human owner" or "main agent"
#                 (never a worker; workers execute, they do not own rows)
#   R5. custody   who physically holds the work right now (main agent|worker id)
#   R6. updated   stamped automatically by the tool — do not pass it
#   R7. source    where this row came from: "owner chat ruling", "commit <sha8>",
#                 "review <sha8>", "audit <date>" — always traceable
#   R8. summary   the deliverable + the verification, stated so a reviewer can
#                 re-run it. Cite commits shas8 and gate ids where they exist.
#                 Bare 8-hex tokens outside a role-vocabulary context trip
#                 the 8-hex governance probe — cite commits as "commit <sha8>".
#
# WORKED EXAMPLE (from STATE-20260901-011, a real closed row):
#   id: STATE-<date>-<nnn>
#   kind: checkpoint | custody | decision | hazard | lane
#   status: open
#   owner: main agent
#   custody: main agent
#   updated: 2026-09-01
#   source: euromonitor-only branch taxi removal (commit 458b1c1)
#   summary: >
#     Taxi dataset removed from the euromonitor-only branch: 128 project/ taxi
#     files deleted (6 experiment series, taxi configs, taxi code/tests),
#     11 taxi gates removed. LOCAL-CI GREEN (parity/ruff/mypy/vulture/configs/
#     shell/pytest/project-tests).
#
# ANTI-PATTERNS (seen in the wild, rejected):
#   - summary that cites no verification ("done" is not a summary)
#   - owner set to a worker subagent
#   - "landed <sha>" claimed while the work sits uncommitted
#   - closing a row whose hazard still exists (close=archive, not delete)

# The template's JSON scaffold — paste into `record add` args ("updated" is
# tool-stamped; omit it):
{
  "id": "STATE-YYYYMMDD-NNN",
  "kind": "lane|custody|decision|hazard|checkpoint",
  "status": "open",
  "owner": "human owner|main agent",
  "custody": "main agent|<worker id>",
  "source": "<provenance: owner chat ruling / commit sha8 / review sha8>",
  "summary": "<deliverable + verification, re-runnable by a reviewer>"
}
