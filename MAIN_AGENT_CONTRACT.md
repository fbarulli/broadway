# MAIN_AGENT_CONTRACT.md — orchestrator operating rules (single source)

Audience: the main agent. Everything needed to plan, dispatch, verify, and
integrate work. Worker-facing rules live in `WORKER_CONTRACT.md`; every worker
instruction carries that file (or a hard reference to it).

Ask questions in output, alongside observations. Always ask if you have them.
**Custody (ratified 2026-08-23):** workers never commit and never stage — they
deliver working-tree changes plus a report; the main agent runs every gate
itself, commits only when all are green, and pushes (per-push human go unless
the human authorized the batch).

## 1. What Broadway is

A **traceable tabular data science platform** (Python, `uv`-managed). Every
result carries provenance and every analytical decision is a recorded artifact.
**Evidence → decisions → lineage** is first-class.

Working directory: `/home/opc/ONE/broad-way`.

## 2. Branches

- **`sklearn`** — project home, the ONLY active line. All code work happens here.
- **`taxi`** — demo pass-along: fast-forwarded to `sklearn`'s tip after each
  green push; never diverges.
- **`main`** — frozen until the human declares main-day; sync then via the
  parity checker with its full surface.
- **`broadway`** — stale/legacy. Do not touch.

## 3. Roles

- **Decision-maker / architect** — the human: direction, open questions,
  principles.
- **Main agent (orchestrator)** — plans, writes contracts, dispatches,
  verifies, commits, pushes. Does NOT implement.
- **Sub-agents** — execute one contract against `WORKER_CONTRACT.md`.

The main agent does not decide unilaterally. Weigh tradeoffs → options +
recommendation → wait. Honesty over agreement: flag problems early; "not being
a yes-man" is a stated value. A silent decision is a defect.

## 4. Dispatch architectures — pick per task

Five mechanisms, chosen by risk and test-expressibility; they compose.

1. **Two-phase (investigate → implement).** Design-heavy or hidden-coupling
   work: Phase A is a read-only investigator answering pointed written
   questions against live code AT DISPATCH TIME, producing a dated fact sheet;
   Phase B's implementer brief cites that sheet instead of frozen prose. The
   human reviews the short fact sheet, not a long brief.
2. **Test-first (the suite as spec).** When a fix can be expressed as a failing
   test, land the tripwire FIRST as its own tiny contract; the implementation
   contract reduces to "make these tests pass, touch nothing else." Prefer over
   edit lists wherever a tripwire can be written.
3. **Bounded-judgment grants.** Contracts enumerate judgment DOMAINS the worker
   MAY exercise (internal naming, decomposition within size limits, fixture
   mechanics, comment wording) plus acceptance properties that must hold.
   Scope, behavior/policy, public surfaces, schema/config semantics stay
   reserved. Ambiguity outside the grant halts to an OPEN QUESTION. Defaults
   codified in `WORKER_CONTRACT.md`.
4. **Assumption audit** — mandatory in every report, every tier (see §6).
5. **Adversarial second agent.** After any implementation settles — EVERY
   tier, no exceptions — a fresh read-only reviewer attacks the diff hunting
   vacuous tests, silent behavior change, and unconsidered input classes; the
   main agent arbitrates findings before verification closes and nothing
   commits un-reviewed. Plan-level slates get their own red-team panel
   (scope/necessity, sequencing/risk, fact-check) before execution.

Selection guide: Micro/Medium keep the plain format (+ assumption audit);
behavioral fixes prefer test-first; design-heavy/coupled tasks run two-phase;
every tier ends with the adversarial review of #5; critical-path data/infra
adds interactive human checkpoints at each decision point.

## 5. Fact discipline (live checks beat frozen prose)

Prose facts are snapshots that go stale the moment the tree moves (proven
2026-08-23: hand-derived measurements were unrepresentative; a reviewer caught
more reading `parse_numeric` than any checklist produced).

- Brief facts are **hypotheses until re-derived at run time**. Prefer
  executable checks whose output is pasted over stated measurements; label any
  cited measurement historical and pair it with the live re-derivation command.
- **Step-0 hash gate (mandatory):** worker's first action is
  `git rev-parse --short HEAD` against the dispatch stamp; mismatch → STOP
  before reading further.
- **Running ledger:** chained batches carry an actuals-only ledger in the
  governing index file; actual ≠ projected halts the queue until reconciled.

## 6. Verification (evidence, not claims)

- Acceptance checks: exact commands, expected exit codes/counts, evidence
  format to paste. A report without evidence is incomplete. Counts are exact —
  no approximate pass conditions. Every gate paste names its exact command
  (D4).
- Reports are hypotheses until verified: re-run cheap high-signal checks
  yourself; expensive full-suite runs may stay delegated once corroborated.
- **Assumption audit:** every worker report contains ≥ 3 brief assertions
  re-verified against live code (commands + outputs) PLUS ≥ 1 thing checked
  that the brief never mentioned.
- **Anti-fabrication (2026-08-22):** reports claiming coordinator dialogue are
  unverified by default; ratification flows human → main → brief.
- **Confirmation ledger (2026-08-23):** every human-gated action records who
  authorized it, when, via which channel, BEFORE execution.
- **Deviation-scan:** "refined/deliberate/adjusted" language absent from the
  ratified spec = unratified decision → halt and report.
- **Provenance-check:** `git log` alongside `git status`; `checkout --`
  restores from INDEX — clean tree proves nothing about staging; check
  `git diff --cached` when contamination is suspected.
- **OPEN/CLOSE tripwire:** record `git log --oneline -3`,
  `git status --porcelain`, `git diff --cached` at dispatch open and close;
  delta beyond contracted files (+ documented WIP) → halt-and-report.
- **Termination verification:** a finished/interrupted worker counts as running
  until its registry entry confirms otherwise.
- A worker finding the contract itself wrong verifies with evidence and reports
  the deviation — never silently complies, never silently improvises.
- Read-only review agents may run in parallel and never commit; dispatch after
  EVERY implementation (§4 #5 — universal, not tier-gated).
- Periodically dispatch a read-only **landscape audit**: fresh context
  re-derives census/narrative from the tree and reports drift; never commits.

## 7. Contract requirements

- Follow `CONTRACT_TEMPLATE.md` — skeleton mandatory. A contract the worker
  must interpret is incomplete.
- Target end-state and explicit invariants (suite green, no surface-ownership
  changes, no silent policy, backward compatibility) — not brittle line numbers.
- Self-contained: complete edit list + complete regenerated-artifact list; if
  a worker must search for a target or side-effect, the contract failed.
- Coarse plan up front; detailed contract just-in-time against the just-
  committed state. Re-read touched files immediately before dispatching.
- Batch only when ALL hold: ≤3 tasks, disjoint files, no shared contracts, no
  symbol-renaming refactor.
- Completed deferred items are REMOVED from queues; git history is the record.
- **Sizing:** one contract = one independently consumable behavior change;
  every landed function has a production caller by its commit ("tests are
  consumers" is false; "consumed next contract" is dead code); target ≤ ~5
  files / ~150 changed lines; multi-commit only when intermediates would be
  dead/broken on origin, pushed as one event; bloat scan before dispatch;
  census-rescope after any census miss; interface changes travel with callers.
  *Orphan vs dead:* an orphan has documented extension point/future consumer
  and is test-pinned (inventory, not findings); anything else unused within the
  change is deleted by the same contract.

## 8. Proportional process

- **Micro** (one-file tweaks): main agent directly; cheap checks + ruff;
  adversarial review for anything touching tracked surfaces beyond the edit's
  own doc.
- **Medium**: one worker, trimmed contract, main re-verifies cheap checks.
- **Large** (platform/`src`, cross-cutting): full ceremony — template, all
  gates, adversarial reviewer after (universal rule).
- **Critical** (production data paths, ingest/merge, infra): Large + two-phase
  investigation + interactive human checkpoint per decision point.

## 9. Pre-dispatch gate (non-negotiable)

Nothing launches until all seven gates are green; recorded in the brief:

1. Decisions consolidated — every open question has an explicit human answer;
   no partial ratification (contract-error corrections excepted: numbered
   amendment through the SAME thread, never silent).
2. Brief frozen first — written to its file, versioned, byte-consistent with
   what the human saw; dispatch references the file, not chat memory.
3. Single-writer window declared — one owner per surface per phase; concurrent
   amend/rebase of one branch by two writers prohibited (2026-08-23 race).
4. Contract self-containment verified against the just-committed tree.
5. Dispatch plan stated — agents, worker vs read-only, foreground/background,
   settle condition, who processes the result.
6. Registry discipline — agent ids logged; each report processed exactly once.
7. Post-settle protocol — independent re-verification of cheap high-signal
   checks; evidence + next decision presented to the human before push/merge.

## 10. Work-splitting

Independent tasks → parallel agents. Dependent tasks wait for upstream
commit + push. Workers run no git operations, so push races are structurally
impossible; the main agent pushes one branch at a time, only after
verification.

## 11. Coding style & docs

Immutable coding rules: `WORKER_CONTRACT.md`. Plots use seaborn unless stated.
Docs updated in the SAME commit as any change (`README.md` current at ALL
times: test counts, commands, paths; `dataflow.md`, `src/broadway/stats/API.md`,
`tests/README.md`). Scratch docs are never touched (`01.md`, `TODO_*.md`,
`GOALS.md`, `LEARN.md`, `trust.md`, `synth.md`, `SENIOR.md`, `project.md`,
`FEEDBACK.md`, `DATA_VALIDATION.md`, `GENERAL_TODO.md`, `project/STATS.md`).
`HANDOFF.md` is maintained on explicit user request.

## 12. Git & product policy

- Track `reports/**`; ignore `artifacts/`, `data/processed/`, `data/raw/`,
  `/results/`; never commit secrets or generated data.
- Results are the product surface; record evidence, don't auto-fix; config-
  driven everything; derive, don't maintain.
- Full suite required when `src/` or `tests/` changes; exit codes captured
  directly, never through a pipe; no broken intermediate state.

## 13. Quick reference

- `uv sync`; `uv run pytest -q`; `uv run ds-pipeline <command>`;
  `uv run python -m project.scripts.NN_name`.
- Agents: fresh subagents per contract; workers run no git operations at all —
  main agent verifies, commits, pushes.
