# LIVE_VERIFICATION.md — evidence on real data (SSOT)

Owner: verification evidence.

The factory accepts behavior only when it has been observed on real data. A passing
unit test proves a function; it does not prove the step produced the right artifact.

## 1. What counts as live data

The dataset named by the active config:

- the canonical parquet under `data/processed/` (or the configured equivalent), or
- the configured sample under `data/samples/` when the step is scoped to a sample.

Not live data: an ad-hoc frame built in a REPL, a test fixture, a mock, a synthetic
matrix. Those are valid at level 1 only (§2).

## 2. Verification ladder

| level | proves | allowed data |
|---|---|---|
| L1 — unit | a pure function's contract | synthetic or fixture |
| L2 — step | the step runs end to end and writes its artifacts | **live** |
| L3 — artifact | values/plots/schema match expectation or a pinned reference | **live** |

A change is verified at the highest level it can affect. Any `src/` change that
produces an artifact requires **L2 + L3**. A refactor with no behavior change still
requires L2 on the affected step.

## 3. Evidence pack

The worker pastes into its report, and main reproduces independently:

1. `git rev-parse --short HEAD` and `git status --porcelain`.
2. The exact command that was run.
3. Dataset name and row count — **provenance, not a validity check** (§4).
4. Wall time, the profile artifact path (`PROFILING.md`), and the log path
   (`LOGGING.md`) for the run.
5. Every output path, plus the key metric values — never just "it ran".
6. An artifact excerpt, diff, or digest proving L3.
7. Every gate command with its exit code.

"Verified" means items 1–7 exist. A claim without the command and its output is
`UNVERIFIED`, and prose proximity never upgrades it.

## 4. Boundary with the data-integrity prohibition

This document adds **no** data-integrity, existence, size, hash, or staleness gate.
Row counts, hashes, and timestamps are recorded as provenance so a result can be
traced; they are never used to accept or reject data. Data changes produce a new
artifact — they are not "checked".

## 5. Determinism

- Seeds come from config; no implicit randomness.
- The same config and data reproduce the same metrics within a tolerance declared in
  config. The tolerance is part of the acceptance check.
- Environment (commit, dependency lock, Python version) is recorded with the run.

## 6. Failure

A live run that fails is reported as a failure. Substituting a smaller, cleaner, or
synthetic frame to obtain green is a violation and is itself the finding. If the live
dataset cannot be run, the acceptance is blocked and the reason is recorded.

## 7. Enforcement

- `CONTRACT_TEMPLATE.md` requires a live-data acceptance check for artifact-producing
  changes.
- The reviewer re-runs the command and compares the artifact; a differing number is a
  defect, not a tolerance to negotiate after the fact.
- The evidence pack is the only accepted form of "done" at station 4
  (`FACTORY.md`).
