# AGENT_WORKER_CONTRACT.md — immutable worker rules

These rules never change. Apply them to every change, every time.

- No hardcoded values.
- ALWAYS present decisions to the user; NEVER decide unilaterally.
- Type hints on all public functions.
- Strategic logging only (stage boundaries, results, errors; never inside loops).
- Catch exceptions only when recoverable; let everything else bubble up.
- YAML = single source of truth: no `get(key, default)`, no hardcoded values.
- ~25-line functions; single responsibility; no dead/noise code.
- Derive, don't maintain: never write state that can be computed at render
  time (no caches, snapshots, or derived-status files) — compute from the
  tree/records instead. The platform derives; it does not store derived state.

## Custody (supersedes any earlier commit-locally language)

Workers run **zero git operations**: no add, no commit, no stash, no branch,
no checkout, never push. Deliver working-tree changes plus a report; the main
agent tests every gate itself, commits when green, pushes on human go.

## Live fact-checking duty (added 2026-08-23)

Facts stated in a brief are hypotheses conditioned on the tree it was written
against — re-derive, don't trust:

- **Step-0 hash gate:** first action of every dispatch is
  `git rev-parse --short HEAD` against the dispatch stamp. Mismatch → STOP
  before reading or acting on anything else in the brief.
- **Re-verify ≥ 3 assertions** from the brief against live code before
  implementing; paste the commands + outputs into your report.
- **Assumption audit (mandatory report section):** list those three
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

