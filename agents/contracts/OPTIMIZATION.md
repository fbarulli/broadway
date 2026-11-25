# OPTIMIZATION.md — profile-driven optimization (SSOT)

Owner: performance optimization law. Measurement itself lives in `PROFILING.md`;
this file owns when and how work is changed for speed.

## 1. Preconditions

Optimization starts only when all hold:

1. pass 1 (single responsibility) and pass 2 (recomposition, `CODE_STANDARD.md` §2)
   are complete;
2. the build is **correct**: L1 tests green and one live L2 smoke written its
   artifacts;
3. a **baseline profile record** exists (`PROFILING.md`) for the surface being
   changed.

Optimizing an unverified build is forbidden — a fast wrong answer is not progress.
Optimization never fixes a bug; a defect found during profiling is routed back to
the build station.

## 2. Profile before changing

Name the hot path explicitly: `file:line`, the metric, its share of total wall
time, and the dataset size. A change without a named hot path is speculative and is
refused.

## 3. Optimization ladder (first rung that wins)

1. **Remove work** — fewer traversals, fewer copies, fewer reads, memoize a value
   already computed.
2. **Use the library's vectorized/batched path** — `groupby.agg`, `transform`,
   `np.where`, sklearn's native multi-output, `polars` expressions.
3. **Batch/chunk IO and memory layout** — read columns actually used, convert dtypes
   once, avoid `object` columns.
4. **Parallelism** — only with the configured worker count; never an inline
   `Pool()` or unconfigured thread count.
5. **Micro-optimization** — last, and only when the profile proves the line matters.

Each rung is tested and measured before moving to the next. Do not stack unmeasured
changes.

## 4. Benchmark protocol ("runs to optimize")

- Warm up once, then run the configured N repetitions.
- Report **p50 and p95**, the variance, the environment, the dataset and its row
  count, and the profiler overhead.
- Same config, same data, same machine for before/after. A single run is not
  evidence.
- One change per measurement, so the delta is attributable.
- The noise band and N live in config. A change inside the noise band is a tie:
  prefer the simpler code.

## 5. Keep or revert

Keep an optimization only when the measured delta exceeds the noise band and the
surface stays within budget. Otherwise revert it — dead complexity is a cost even
when it is fast.

Record for each kept change: baseline path, hot path, the change, the before/after
p50/p95, and the resulting share of total time.

## 6. Output equivalence

The optimized build must produce the same artifacts and metric values within the
declared tolerance (`LIVE_VERIFICATION.md` §5). A different number is a behavior
change: stop, route it to the spec, and re-verify — do not absorb it into an
"optimization" commit.

## 7. Stopping rule

Stop when the surface is within budget, or after the configured number of
optimization iterations. Record the remaining gap explicitly as a waiver with its
cost; never mark an over-budget surface as passing.

## 8. Forbidden

- Speculative optimization without a profile record.
- Behavior changes hidden in an optimization commit.
- Hand-rolled parallelism outside the configured workers.
- Adding a heavy native dependency (numba, Cython, a GPU runtime) without a ruling
  and a registry row (`CODE_STANDARD.md` §4).
- Optimizing private helpers while an obvious algorithmic waste remains in the
  step.

## 9. Enforcement

- Profile gate compares against the pinned baseline (`PROFILING.md` §5).
- Output-equivalence check on the live artifact, within tolerance.
- Reviewer confirms the hot path was measured, the delta exceeds noise, and no
  behavior changed.
