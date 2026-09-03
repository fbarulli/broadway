# TEMPLATE — STATE terminal disposition (close or void)
#
# Close a row when its deliverable is LANDED and verified, void when it was
# created in error or is superseded by a better row. Use the ONLY sanctioned
# terminal path (never hand-edit STATE.md or the archive):
#
#   uv run python agents/tools/state_records.py record close --id <id> \
#     --reason "<evidence sentence>"
#   uv run python agents/tools/state_records.py record void --id <id> \
#     --reason "<evidence sentence>"
#
# REQUIREMENTS (enforced by state_records.py close/void):
#   R1. The row must exist and be non-terminal (open/blocked/approved).
#   R2. reason must state the EVIDENCE, not the intention:
#       landed commit sha8 + the verification that proves it (e.g. "removed
#       in 458b1c1, LOCAL-CI GREEN") — a reviewer re-runs the verification.
#   R3. close ARCHIVES (never deletes): the row moves to
#       agents/ledger/archive/<YYYY-MM>.md with a STATE-ARCHIVE:<id> marker;
#       git history keeps everything either way.
#   R4. A hazard closes only when the hazard is FIXED, not when it becomes
#       familiar; a lane closes only when its deliverable landed green.
#   R5. Closing does not update gates.yaml — if the closed row's surface had
#       a gate row, re-anchor or retire that row in the same batch.
#
# WORKED EXAMPLE (real close, STATE-20260901-011):
#   record close --id STATE-20260901-011 \
#     --reason "taxi removal landed in commit 458b1c1, LOCAL-CI GREEN"
#
# SCAFFOLD — fill, then run the command above:

id: STATE-YYYYMMDD-NNN
disposition: close|void
reason: >
  <deliverable> landed in commit <sha8>; verification: <exact command + result>.
