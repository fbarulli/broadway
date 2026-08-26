# WORKER_CONTRACT.md — immutable subagent rules

Audience: every dispatched subagent (implementer, investigator, reviewer).
One turn, one contract, then stop and report. These rules never change.

## Custody

Zero git operations: no add, no commit, no stash, no branch, no checkout,
never push. Deliver working-tree changes plus a report; the main agent tests
every gate itself, commits when green, pushes on human go.
Exception: read-only git inspection (rev-parse/status/diff/log) is allowed and
expected; write operations (add/commit/stash/branch/checkout/push) never.
Siting law: scratch files live OUTSIDE the repo (mktemp -d) or not at all —
never `.wg2_scratch/`, `.tmp-*/`, or cache dirs inside the tree; the main
agent deletes on sight and repeat offenses are a lane-failure class.

## Immutable coding rules

- No hardcoded values.
- ALWAYS present decisions; NEVER decide unilaterally (see grant below).
- Type hints on all public functions.
- Strategic logging only (stage boundaries, results, errors; never inside loops).
- Catch exceptions only when recoverable; let everything else bubble up.
- YAML = single source of truth: no `get(key, default)`, no hardcoded values.
- ~25-line functions; single responsibility; no dead/noise code.
- Derive, don't maintain: never store state that can be computed from the
  tree/records at render time. The platform derives; it does not store
  derived state.

## Live fact-checking duty

Facts stated in a brief are hypotheses conditioned on the tree it was written
against — re-derive, don't trust:

- **Step-0 context gate:** before anything else, read
  `agents/ledger/STATE.md` (lanes in flight, hazards, open arbitrations)
  AND derive landed facts from `git log --oneline -12` + `git status` —
  STATE.md deliberately does NOT mirror git; where they appear to
  disagree, git wins. If instructions contradict either — HEAD moved, an
  assigned file changed hands, a lane believed quiet is active — STOP and
  report "stale-on-arrival" with the exact contradiction. Never improvise
  around drift.

- **Step-0 hash gate:** first action of every dispatch is
  `git rev-parse --short HEAD` against the dispatch stamp. Mismatch → STOP
  before reading or acting on anything else in the brief.
- **Re-verify ≥ 3 assertions** from the brief against live code before
  implementing; paste the commands + outputs into your report.
- **Assumption audit (mandatory report section):** those three
  re-verifications PLUS at least one thing you checked that the brief never
  mentioned — a surprise, a neighboring hazard, an input class nobody named.
  "none" is almost never the honest answer.

## Bounded-judgment grant (default domain)

Unless the contract explicitly narrows it, you MAY decide without asking:
internal naming of new private symbols, function decomposition within the
~25-line rule, test fixture mechanics that preserve the asserted properties,
and comment wording in your own style. You may NEVER decide: scope, behavior
or policy changes, public surface or schema/config semantics, another owner's
surface, or anything a failing acceptance check depends on. Ambiguity outside
the grant → OPEN QUESTION in the report; an undisclosed decision is a
violation, an unanswered question is not.

## Mandatory gates (single vocabulary)

Every contract whose diff touches `src/`, `tests/`, `scripts/`, `configs/`,
`.github/`, or any CI-lint surface MUST run `bash scripts/run_local_ci.sh`
(doc-only edits may pass `--static`) and paste the five PASS/FAIL banners in
its report. pytest-only gate lists are INVALID for code-bearing contracts —
lint (ruff), types (mypy), config-parse, and the coverage floor live ONLY in
that script; assembling gates ad hoc is how F401-class residue reached
remote CI (incident GATE-SSOT, FIXES.md). A full-suite pytest run alone does
not substitute.

## Report format (every dispatch)

1. Step-0 gate result (stamp vs actual HEAD).
2. Per-file change summary with line references.
3. Acceptance-check pastes demanded by the contract (exact commands, outputs,
   counts — before/after where applicable). Paste the COMMAND alongside every
   output tail (D4: an honest number under a different invocation is still a
   mismatched gate).
4. **Assumption audit** section (above).
5. **OPEN QUESTIONS** section — mandatory; write "none" only if genuinely
   nothing surfaced. An unanswered question is correct behavior; an undisclosed
   decision is a violation.
