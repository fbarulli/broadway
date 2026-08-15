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
