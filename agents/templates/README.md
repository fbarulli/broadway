# agents/templates — the sanctioned TEMPLATE SET for every governed step

One template per workflow step. An agent does NOT free-hand these artifacts;
it copies the template and fills the placeholders. Requirements for each
template live in the template itself (the REQUIREMENTS block) so the rule and
the scaffold can never drift apart.

Determinism law (2026-09-03): any artifact with a template must be created
FROM the template. Free-handing is a lane violation.

| step | template | enforcement |
|------|----------|-------------|
| 1. STATE row (open a lane) | state-row.md | `state_records.py record add` validates columns |
| 2. commit message | commit-msg.md | `scripts/ledger_commit.sh` gates before git commit |
| 3. gates.yaml row (touch a governed surface) | gate-row.md | registry law: touched surface must own a row |
| 4. STATE terminal (close/void) | state-close.md | `state_records.py record close/void` validates |
| 5. push | — (ship.sh usage header IS the template) | `scripts/ship.sh` gates the batch |

Usage per template is documented inside each file.
