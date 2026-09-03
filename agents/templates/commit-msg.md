# TEMPLATE — commit message
#
# Copy this scaffold, fill it, commit via the gated front door:
#
#   bash scripts/ledger_commit.sh -   # reads message from stdin
#
# REQUIREMENTS (gated by scripts/ledger_commit.sh BEFORE git commit runs):
#   R1. Subject: TYPE(scope): summary — one line, imperative, no period.
#       TYPE is one of the ratified vocabulary words found in history:
#       FEAT | FIX | STATE | GOVERNANCE | ANALYSIS | EVAL | CHORE | REGISTRY
#       | STYLE | REFACTOR
#   R2. Body: what changed, why, and the exact verification performed.
#       Every claim must be one a reviewer can re-run (command + result).
#       No "trust me" claims.
#   R3. Tier trailer: `Tier: FULL|FAST|STATIC|DOCS` — case-sensitive, exactly
#       one, from D34/D35 ratified grammar (validated by tier_gate.sh).
#   R4. Reviewer trailer: required when Tier is FULL — an 8-hex EVENTS verdict
#       id that resolves as a row id or authority registration at HEAD.
#   R5. State trailer: `State: STATE-YYYYMMDD-NNN` — the CURRENT row this
#       commit advances. The row must exist in agents/ledger/STATE.md at
#       commit time (bidirectional linkage: commit -> row, row -> commit).
#
# WORKED EXAMPLE (real history, f0dcb87's shape):
#   FEAT(euromonitor): consolidated prep+encode+link pipeline + notebook
#   - 08_pipeline.py: one re-runnable command (dedupe -> encode -> resolve).
#   - _link.py: linking logic extracted from notebook into shared
#     resolve_items() — single source of truth.
#   - Pipeline result: 71,623 SKUs -> 21,781 ITEMs (verification re-run:
#     bash scripts/run_local_ci.sh -> GREEN).
#
#   Tier: FULL
#   Reviewer: 39de4245
#   State: STATE-20260902-016
#
# SCAFFOLD — fill every line:

TYPE(scope): one-line summary of the deliverable

- <what changed, file by file>
- <why: the ruling or finding this answers>
- <verification: the exact command run + its result>

Tier: FULL|FAST|STATIC|DOCS
Reviewer: <8-hex verdict id — required iff Tier FULL>
State: STATE-YYYYMMDD-NNN
