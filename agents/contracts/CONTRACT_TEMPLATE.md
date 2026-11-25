# CONTRACT_TEMPLATE.md — mandatory dispatch skeleton

## Sizing (D20 — hard caps)
MICRO ≤2 files / ≤30 lines / 1 deliverable · MEDIUM ≤4 files / ≤150 lines /
≤3 deliverables / ≤25 min · anything larger DECOMPOSES into MEDIUMs before
dispatch. INVESTIGATE (read-only fact sheet) is a separate contract from
EXECUTE (consumes the sheet verbatim, pasted into the prompt). Parallel only
when custody-disjoint AND output-independent; default sequential. Heartbeats
from launch on >20-min dispatches.


Every agent contract fills this skeleton before dispatch (MAIN_AGENT_CONTRACT.md
§7). A contract the worker has to interpret is incomplete — it must
EXECUTE, not explore.

## Task

One or two sentences: the goal, the target end-state, and the scope boundary
(what is explicitly NOT in scope).

## Edit list (complete)

For every file that changes:
- path: exact current content → exact replacement (or the precise end-state
  for new files), plus every regenerated artifact the command sequence
  rewrites.
- If the worker would have to search to locate a target or a side-effect, the
  contract is incomplete.

## Blast radius (mandatory, before any implementation reasoning)

For every touched surface: `bash scripts/blast_radius.sh <path> [symbol]` —
paste the governance (gates/owners/tests) AND structural (graphify reverse
traversal: real callers with file:line) output here. A contract without this
evidence is incomplete: the worker must not reason about changes it has not
measured. Reviewers re-run the same command (REVIEWER_CONTRACT.md §4).

## Constraints

- Immutable worker rules: `WORKER_CONTRACT.md` (type hints on public
  functions, strategic logging only, catch only recoverable exceptions, YAML
  single source of truth, ~25-line single-responsibility functions, no dead
  code; step-0 hash gate; assumption audit; bounded-judgment grant).
- Custody: workers run no git write operations. Exception: read-only git inspection
  (rev-parse/status/diff/log) is allowed and expected; write operations
  (add/commit/stash/branch/checkout/push) never.
- Invariants: `bash scripts/run_local_ci.sh` green (parity, ruff, mypy,
  configs, pytest+cov≥95% — the same gates ci.yml runs via the same script);
  no surface-ownership changes; no silent policy; backward compatibility.
- OPEN/CLOSE tripwire: `git log --oneline -3` + `git status --porcelain` +
  `git diff --cached` (must be empty) recorded at dispatch open and close;
  any delta beyond the contracted files → halt-and-report.
- Files NOT listed in the edit list must not change.

## Engineering laws (mandatory — a blank field is an incomplete contract)

- **Library decision** (`CODE_STANDARD.md` §3): name the external library or existing
  helper used, or state why rungs 1–3 fail and new code is required.
- **Recompose pass** (`CODE_STANDARD.md` §2): what pass 2 fused or vectorized
  (traversals, loops, copies, IO), or why the pass-1 code already traverses once.
  "None" without that evidence is an incomplete contract.
- **Consolidation check** (`CONSOLIDATION.md`): name the canonical owner of every new
  behavior and the existing implementation searched. A near-duplicate fails review.
- **Profile budget** (`PROFILING.md`): tiers touched, metric, budget value, artifact
  path. No budget is written as an explicit waiver naming the cost — never left blank.
- **Optimization evidence** (`OPTIMIZATION.md`): baseline profile path, hot path
  `file:line`, the change, and the before/after p50/p95; or "within budget, no
  optimization attempted".
- **Logging impact** (`LOGGING.md`): new/changed log lines and their level, or
  "none"; no `print` in library code; the `run_id` correlation path for the run.
- **Dependency impact** (`CODE_STANDARD.md` §4): new/removed dependency, its registry
  row in `configs/dependencies.yaml`, owner, scope, and adapter seam.
- **Live-data check** (`LIVE_VERIFICATION.md` §3): for artifact-producing changes,
  dataset, row count, command, and the metric/artifact to compare. L1-only acceptance
  is allowed only for pure functions.

## Acceptance checks (evidence format)

Numbered; each is an exact command + expected result + what the worker pastes
back (exit codes, counts, diffs, git status). Prefer structural checks over
substring greps when the artifact is structured (e.g. for HTML: assert the JS
sits between `<script>` tags, not merely that its text appears somewhere).
At least one check must execute the live-data command from the row above and
compare an artifact (§3 evidence pack) — a suite of unit checks alone does not
satisfy the contract for artifact-producing changes. When pass 2 changes a hot
path, one check must also compare the optimized artifact against the pass-1
result within the declared tolerance (`OPTIMIZATION.md` §6).

## Commit (main agent ONLY) — workers never touch git

- Branch: the contracted branch (e.g. `sklearn`). One commit,
  message: `<…>`. The worker performs NO git operations — it delivers
  working-tree changes and its report; a worker report claiming it committed
  or pushed is itself a deviation.
- Evidence the main agent verifies before committing: full diff vs the edit
  list, gate outputs re-run independently, `git status --porcelain`,
  `git diff --cached` (empty until the main agent stages).
- The main agent commits only when ALL gates are green, then pushes on human
  go (`git push origin <branch>`, no force); external failures are reported,
  never force-pushed.
