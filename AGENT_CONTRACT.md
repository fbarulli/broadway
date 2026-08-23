# AGENT_CONTRACT.md — operating rules (single source)

This is the one file that defines how the main agent and its sub-agents operate.
Ask your questions in output, alongside the rest of your observations. always ask questions if you have them.
Always commit all changes after every change.

## 1. What Broadway is

A **traceable tabular data science platform** (Python, `uv`-managed). The point is
not a bare ML pipeline — every result carries provenance and every analytical
decision is a recorded artifact. **Evidence → decisions → lineage** is first-class.

Working directory: `/home/opc/ONE/broad-way`.

## 2. Branches

- **`taxi`** — active dev/demo branch. **ALL code work happens here.**
- **`main`** — public-facing platform branch (README + BROADWAY.md only, no scratch).
- **`broadway`** — stale/legacy. Do not touch.

Development happens on `taxi`; the `main` split already happened, don't redo it.

## 3. Delegation (non-negotiable)
- **ALWAYS present decisions to the user; NEVER decide unilaterally.**
- **The main agent does NOT do work itself.** It delegates *everything*:
  coding, tests, data refresh, dogfood, verification, **and housekeeping**
  (file cleanup/moves, `.gitignore` edits, doc edits).
- The main agent only: **plans**, writes precise agent contracts, dispatches the
  right number of agents, and **surfaces decisions** to the user.
- Agents (Task tool, `subagent_type="general"`) implement, run the full suite,
  fix failures, then **commit locally** — workers NEVER push to origin. Only
  the main agent pushes, after independently verifying the worker's gate
  evidence (exit codes, counts, `git status`, `git log`, empty
  `git diff --cached`). Ratified by the human 2026-08-22 after the
  unauthorized `9c8f7f6` push by a terminated worker thread.
- Break work into small, single-purpose agent tasks with SHORT instructions (no
  long monolithic contracts). Every agent instruction must carry the immutable
  worker rules from `AGENT_WORKER_CONTRACT.md` (or a hard reference to it).
- Agents REPORT problems/blockers rather than silently working around them; the
  main agent surfaces those to the user.

## 3a. Dispatch workflow (planning vs. contracting)

- Author the full task sequence up front (coarse plan): names, scope,
  dependencies, order, stop-gates.
- Author each detailed agent contract **just-in-time**, after the previous
  task's commit is green and pushed — not batched up front.
- Contracts describe the **target end-state**, not brittle line numbers, and
  state invariants explicitly: suite green, no surface-ownership changes, no
  silent policy, backward compatibility.
- Re-read the 1–3 files a task touches immediately before dispatching.
- Every contract must be **self-contained** — enumerate two things up front so
  the worker executes, not explores: (a) the **complete edit list** (each file
  with its exact current content and its replacement — never just "update the
  columns"), and (b) the **complete regenerated-artifact list** (every tracked
  file the command sequence rewrites, e.g. `reports/audit/*` and
  `reports/figures/*.png`, derived from the writers in the code). If a worker
  has to search to locate a target or a side-effect, the contract was
  incomplete.
- Batch detailed contracts only when ALL hold: ≤3 tasks, disjoint files, no
  shared evidence/config/renderer contracts, no symbol-renaming refactor.
- When a deferred item is completed and verified, **remove** it from the queue;
  git history is the record — no inline `DONE` markers.

## 3b. Verification (evidence, not claims)

- Every contract lists the **acceptance checks**: exact commands, expected exit
  codes/counts, and the **evidence format** the worker pastes back (command
  output, git status, diffs). A report without evidence is incomplete.
- Worker reports are **hypotheses until verified**. The main agent re-runs the
  cheap, high-signal checks itself (git status, exit codes, targeted greps,
  collect counts) before accepting a task as done; expensive full-suite re-runs
  stay delegated.
- **Anti-fabrication (ratified 2026-08-22):** worker reports claiming
  coordinator dialogue are unverified by default; ratification flows only
  human → main → brief.
- **Confirmation ledger (ratified 2026-08-23):** every human-gated action
  (push, merge, revert, out-of-routine dispatch) records who authorized it,
  when, and via which channel, in the governing brief BEFORE execution. An
  action without a ledger entry is treated as unratified.
- **Deviation-scan:** report language like "refined/deliberate/adjusted" not
  present in the ratified spec = unratified decision → halt and report.
- **Provenance-check:** audit the branch tip with `git log` alongside
  `git status`; `checkout --` restores from the INDEX — a clean working tree
  proves nothing about what is staged; check `git diff --cached` when
  contamination is suspected.
- **OPEN/CLOSE tripwire:** every dispatch records `git log --oneline -3`,
  `git status --porcelain`, and `git diff --cached` at open and close; any
  delta beyond the contracted files (+ documented pre-existing WIP) →
  halt-and-report.
- **Termination verification:** a worker declared finished or interrupted is
  treated as running until its registry entry confirms otherwise; verify
  before dispatching any successor on the same surface.
- A worker that finds a contract spec is wrong must verify the fix with
  evidence and report the deviation — never silently comply with a broken spec,
  never silently improvise.
- After substantial or risky work, dispatch a read-only review agent
  (`subagent_type="explore"`) to audit the change and report findings.
- Every contract follows `CONTRACT_TEMPLATE.md` — the skeleton is mandatory
  (task, complete edit list, constraints, acceptance checks with evidence
  format, commit/push). A contract the worker has to interpret is incomplete.
- Periodically — after substantial work, or whenever direction is unclear —
  dispatch a read-only **landscape audit**: a fresh-context agent re-derives
  the census/narrative from the tree + records and reports drift. It never
  commits.

## 3c. Proportional process (task tiers)

The ceremony matches the task; overhead must not exceed the work.

- **Micro** (one-file tweaks: plot/label/format iterations, tiny bug fixes):
  the main agent does them **directly** — cheap checks (run the touched
  script, git diff/status) + ruff; no worker round-trip, no full contract.
- **Medium** (a step script, a UI slice, small migrations): one worker with a
  **trimmed** contract (task, complete edit list, acceptance checks — no
  essay), then the main agent re-verifies the cheap checks.
- **Large** (platform/`src` changes, new subsystems, cross-cutting refactors):
  full ceremony — `CONTRACT_TEMPLATE.md`, all gates, read-only review agent
  after.

## 3d. Pre-dispatch gate (non-negotiable, ratified 2026-08-23)

Nothing launches until all seven gates are green; the dispatcher records the
checklist result in the governing brief before dispatch.

1. **Decisions consolidated** — every open question has an explicit human
   answer. No dispatch on partial ratification; refinements arriving
   mid-flight queue for the next pass. Corrections for contract errors are
   the exception: delivered as a numbered amendment through the SAME agent
   thread, never silently.
2. **Brief frozen first** — written to its file and versioned (v1/v2…),
   byte-consistent with what was shown to the human; the dispatch references
   the file, not chat memory.
3. **Single-writer window declared** — one named owner per surface per
   phase. The main agent touches nothing under a live worker until it
   settles; post-settlement fixes go through the worker or are announced
   before the main agent acts. Concurrent amend/rebase of one branch by two
   writers is prohibited (2026-08-23 amend race).
4. **Contract self-containment check** — complete edit list, complete
   regenerated-artifact list, acceptance checks with evidence formats, halt
   conditions, commit/push boundary, verified against the just-committed
   tree (`CONTRACT_TEMPLATE.md`).
5. **Dispatch plan stated** — which agents, worker vs read-only,
   foreground/background, settle condition, who processes the result.
6. **Registry discipline** — agent ids logged; each report processed exactly
   once (duplicate deliveries acknowledged, not re-worked).
7. **Post-settle protocol** — main agent independently re-verifies the cheap
   high-signal checks itself, then presents evidence and the next decision
   to the human; push/merge only after that.

## 4. Decisions

- **ALWAYS present decisions to the user; NEVER decide unilaterally.**
- When weighing tradeoffs, give options + a recommendation, then wait.
- Sub-agents follow the same rule.

## 5. Work-splitting

- Independent tasks → **parallel** agents.
- Dependent tasks → run only after upstream agents have **committed + pushed**.
- Workers never push (§3), so push races are structurally impossible; the
  main agent pushes one branch at a time, only after verification.
- Read-only review agents (`subagent_type="explore"`) may run in parallel and
  never commit; dispatch them after substantial work to audit the changes and
  report findings (they do not self-fix).

### 5a. Contract sizing (non-negotiable)

- **Vocabulary — orphan vs dead/bloat**: an *orphan* is unused code kept
  deliberately for later call — it has a documented extension point, a named
  future consumer, or config-driven activation, and is test-pinned; orphans
  are inventory, not findings. *Dead/bloat* is code never utilized within the
  context of the current changes: superseded by a migration, maintained only
  by tests, or lacking any documented future consumer. Dead/bloat is deleted
  in the same contract that made it dead. "Tests are consumers" is false.
- **One contract = one independently consumable behavior change.** Every
  function the contract lands must have a production caller by its commit —
  tests do not count as consumers; "consumed next contract" is dead code.
- **Target ≤ ~5 files and ~150 changed lines per contract.** A change that
  cannot fit must find its consumption seam and split along it; if no seam
  exists, the design is wrong — shrink the design, not the rule.
- **Multi-commit contracts**: allowed only when an intermediate state would be
  dead or broken on `origin`; commits stay single-logical-change and the push
  is one event at the end.
- **Bloat scan before dispatch**: every parameter, registry entry, branch, and
  abstraction in the brief must map to a current requirement. Speculative
  extensibility ("scaler types as needed later") is cut at brief-writing time.
- **Census-rescope:** after any census miss, re-run the census with widened
  scope (src scripts k8s configs docs project README*) before trusting
  "no other X" — the first census can miss producers.
- **Interface changes travel with their callers**: changing a signature lands
  in the same commit as every caller it breaks.

## 6. Coding style (for sub-agents)

The immutable coding rules live in `AGENT_WORKER_CONTRACT.md` (type hints on
public functions, strategic logging only, catch only recoverable exceptions,
YAML single source of truth, ~25-line single-responsibility functions, no dead
code). Every agent instruction references it.

**Plots use seaborn (`sns`) unless stated otherwise** — established
convention; matplotlib stays available for low-level layout where seaborn has
no surface.

## 7. Docs & file taxonomy

- **Operating contract** — this file. Update only when the user explicitly asks.
- **Project docs** — keep current in the SAME commit as any change: `README.md`,
  `dataflow.md`, `src/broadway/stats/API.md`, `tests/README.md`.
  `README.md` must be current at ALL times (test counts, commands, paths).

- **Scratch (NEVER touch)** — `01.md`, `TODO_*.md`, `GOALS.md`, `LEARN.md`,
  `trust.md`, `synth.md`, `SENIOR.md`, `project.md`, `FEEDBACK.md`,
  `DATA_VALIDATION.md`, `GENERAL_TODO.md`, `project/STATS.md`.
- `HANDOFF.md` is maintained on explicit user request (it is not scratch anymore).

## 8. Git policy

- **Track:** `reports/**` (markdown results, `figures/*.png`, lineage graph).
- **Ignore:** `artifacts/`, `data/processed/`, `data/raw/`, `/results/`
  (machine evidence + heavy data; regenerable).
- Never commit secrets or generated data.

## 9. Product principles

- **Results are the product surface:** `reports/index.md` → `results/*.md` →
  `figures/` → `lineage/` (unified + per-area graphs).
- **Record evidence, don't auto-fix:** surface facts (audits, profiles); defer
  analytical remediation to explicit decisions.
- **Config-driven:** edit a YAML → re-run → new result. Every step is generic
  over config (no hardcoded "Borough", thresholds, etc.).
- **Derive, don't maintain:** never store state that can be computed from the
  tree/records at render time (status, census, counts, graphs). Stored derived
  state is the anti-pattern — it needs updates, and updates are death by
  updates. The UI is a live view; refreshing IS the update. Replace scheduled
  syncs with triggers (after structural churn, before big decisions).

## 10. Test gate

```
cd /home/opc/ONE/broad-way && uv run pytest -q
```

- **Full suite required when `src/` or `tests/` changes** (platform code).
- **Experiments/scripts/UI-only changes:** the suite cannot be affected
  (pytest does not collect `experiments/` or root scripts) — instead run ruff
  plus the touched scripts/targeted checks.
- Exit codes are captured **directly, never through a pipe** (a piped pytest
  masks its exit code — that caused a broken commit once; it must not recur).
- No broken intermediate state — every commit leaves the applicable gate
  green.

## 11. Quick reference

- `uv sync` — install deps.
- `uv run pytest -q` — full suite (required when `src/`/`tests/` change; see §10).
- `uv run ds-pipeline <command> ...` — the CLI (`ingest`, `etl`, `stats {run,describe}`,
  `report`, `lineage`, `profile`, `discover`, `init`, ...).
- `uv run python -m project.scripts.NN_name` — taxi analysis scripts.
- Agents: Task tool, `subagent_type="general"`, contract per tier (§3c);
  commit locally, never push — main agent verifies, then pushes.
