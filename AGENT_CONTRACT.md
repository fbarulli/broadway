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
  fix failures, then **commit + push** to `taxi`.
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

## 4. Decisions

- **ALWAYS present decisions to the user; NEVER decide unilaterally.**
- When weighing tradeoffs, give options + a recommendation, then wait.
- Sub-agents follow the same rule.

## 5. Work-splitting

- Independent tasks → **parallel** agents.
- Dependent tasks → run only after upstream agents have **committed + pushed**.
- If two agents both commit/push, **sequence them** (avoid git races).
- Read-only review agents (`subagent_type="explore"`) may run in parallel and
  never commit; dispatch them after substantial work to audit the changes and
  report findings (they do not self-fix).

## 6. Coding style (for sub-agents)

The immutable coding rules live in `AGENT_WORKER_CONTRACT.md` (type hints on
public functions, strategic logging only, catch only recoverable exceptions,
YAML single source of truth, ~25-line single-responsibility functions, no dead
code). Every agent instruction references it.

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
Must be green after every change.
No broken intermediate state — every commit must leave the suite green.

## 11. Quick reference

- `uv sync` — install deps.
- `uv run pytest -q` — full suite (must be green).
- `uv run ds-pipeline <command> ...` — the CLI (`ingest`, `etl`, `stats {run,describe}`,
  `report`, `lineage`, `profile`, `discover`, `init`, ...).
- `uv run python -m project.scripts.NN_name` — taxi analysis scripts.
- Agents: Task tool, `subagent_type="general"`, precise contract; require
  full-suite green + commit + push.
