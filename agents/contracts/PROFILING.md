# PROFILING.md — measurement and budgets (SSOT)

Owner: performance law.

## 1. What gets measured

Three tiers. A contract names the tiers it touches; silence means tier 1 + tier 2.

- **Tier 1 — pipeline steps only, for now.** Wired for steps that run through
  `pipeline.run` (`etl`, `contracts`, `features`, `stats`, `causal`, `train`,
  `evaluate`, `baseline`, `full`): one record per invocation. Direct CLI entry
  points (`ingest`, `stats`, `walkthrough`, `report`, `discover`, `decide`, …) are
  **not yet wired** — planned.
- **Tier 2 — where a step opts in.** Wired only for functions a step wraps with
  `phase("…")` on the profiling seam; the train step is the first (`load_features`,
  `split`, `model_fit`, `calibration_fit`, `mlflow_log`, `persist`). Every other
  loop/IO/frame-wide function is **not yet instrumented** — planned. This is where
  cost hides.
- **Tier 3 — opt-in: any function under investigation.** Enabled by the profiler for
  one run; not permanent instrumentation.

The "mandatory" scope above is the target contract, not a statement of current
coverage: tier 2 is wired incrementally, one measured step at a time.

Not measured: trivial accessors, config validation, one-line helpers, and anything
with no loop, IO, or frame operation. Instrumenting those adds overhead and noise
without adding signal — that is a defect under the harness contract.

## 2. One seam

Exactly one instrumentation module (the profiler seam) provides the context
managers: `timed()` yields a wall-clock reader, and `phase(name)` records a named
wall+CPU phase into the active step profile (inert outside one). No `time.time()`,
`time.perf_counter()`, or ad-hoc timers anywhere else in `src/`. A scattered timer
is a consolidation finding.

- Disabled by default and never on a production path.
- Enabled through config/env, not a code edit.
- The profiler's own overhead is measured once per release and recorded with the
  report, so a small regression is never an artifact of the profiler.

## 3. Metrics

Per record: wall time (p50 and p95 over N runs), CPU time, peak RSS, rows processed,
and throughput normalized per 1,000 rows.

Normalization is mandatory: comparing raw seconds across different dataset sizes is
an invalid measurement, not a regression. The dataset size and row count are part of
the record.

## 4. Budgets and artifacts

- Budgets live in config (`configs/tooling.yaml` or the owning step config) — never
  as literals in code.
- Machine records: `artifacts/profiles/<step>.json` (git-ignored, regenerable).
- Published summary, when the result is part of the story: `reports/profiles/<step>.md`,
  rendered from the JSON, never recomputed there.
- The gate baseline is tracked. A new baseline is a deliberate commit with its
  environment and dataset recorded; it is never silently overwritten.
- First run on a surface records the baseline and reports advisory only.

## 5. Gate

A run over budget fails the profile gate. The threshold and the run count N come
from config. Results less than the configured noise band are reported, not gated.

The report shows the raw number, the environment, the dataset and its size, N, and
the profiler overhead. No symbol replaces a number.

## 6. Honesty

- A faster number on smaller data is not an improvement.
- A single run is an anecdote; the gate uses N runs and the configured statistic.
- A missing budget is recorded as missing, not as passing.

## 7. Enforcement

- Tier 1 is wired at `pipeline.run` only; direct CLI entry points are planned.
- Tier 2 is wired at explicit `phase()` call sites, not by blanket instrumentation;
  a new loop/IO added to a measured step carries its own `phase()`.
- Profile gate over the configured surfaces — **planned, not yet wired**; advisory
  until the first baseline exists, blocking after.
- Reviewer checks that new loops/IO in a measured step carry a tier-2 record and
  that no scattered timer was introduced.
- The harness rule "log timing when time is a concern" is satisfied by this seam,
  not by ad-hoc logging.
