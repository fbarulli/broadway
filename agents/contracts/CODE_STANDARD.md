# CODE_STANDARD.md — how code is written (SSOT)

Owner: platform code law.

Referenced, not restated: SSOT, config-over-hardcode, pydantic at boundaries, full
tracebacks, file ≤1000 lines (harness `AGENTS.md`); ~25-line functions, type hints,
YAML SSOT ( `WORKER_CONTRACT.md` ); pattern vocabulary ( `patterns.md` ).

## 1. Shape

- Behavior lives in a class; module level is constants + a thin entry point.
- One public purpose per function; ~25 lines; split when a second branch, loop, or
  responsibility appears. A god-function is a defect, not a style preference.
- Construction lives in a factory (`build` / `from_config`), never scattered at call
  sites. Dependencies are passed in, not instantiated inside (Dependency Injection).
- Types on every parameter and return. Pydantic at every external boundary
  (config, file, network, CLI, env); trust internal types afterwards.
- No dead code, no pass-through wrapper, no "for later" flag. Deletion beats
  addition.

## 2. Two-pass construction (mandatory)

Every nontrivial change is built twice, in the same commit.

**Pass 1 — decompose.** Write the smallest single-responsibility functions. Correct,
typed, testable, library-first (§3).

**Pass 2 — recompose for work.** Before verification, revisit the pass-1 code and
fuse anything that repeats work:

- several passes over the same frame → one pass (`agg`, one `groupby`, one `apply`
  family) when the results share a traversal;
- row-wise Python (`iterrows`, per-row `apply`) → a vectorized expression
  (`np.where`, boolean mask, `map`, `groupby.transform`);
- repeated IO or parquet reads → load once, select once;
- repeated joins/lookups → resolve once, reuse;
- list append loops → comprehension / array construction;
- redundant copies and intermediate frames → select inline;
- the same statistic computed twice → compute once, pass it.

Rules:

- **One responsibility, not one operation.** Single responsibility means one reason
  to change. Removing redundant work is required; merging two responsibilities is
  forbidden.
- **Fuse the loop, not the responsibility.** Keep the pure single-responsibility
  functions; do the combining in the orchestrator or in a single batch/vectorized
  call. Do not inline three responsibilities into one function to save a loop.
- **Correctness is frozen.** Pass 2 must not change results. Any intended behavior
  change is a separate task, not an optimization.
- **Testability survives.** If recomposition makes a function untestable or forces
  mocks, the recomposition is wrong — fuse at the call site instead.
- **Nothing speculative.** "This might be slow" is not a reason to rewrite. Measured
  optimization belongs to `OPTIMIZATION.md`.

## 3. Library-first ladder (mandatory order)

Take the first rung that fully works:

1. **Installed domain library** — scikit-learn, statsmodels, pandas/polars,
   pydantic, mlflow, optuna, scipy, pingouin.
2. **Existing repository helper** — the canonical owner (`CONSOLIDATION.md`).
   Move it to its correct owner if misplaced; never copy it.
3. **Standard library / platform feature.**
4. **New code** — last resort. The contract must state, in one line, which
   library was considered and why it does not fit.

Every hand-rolled algorithm, statistic, encoding, split, metric, or persistence
layer must answer: *which library implements this, and why are we not using it?*
A "domain-specific" answer is not sufficient — the platform is data-agnostic.

Vectorized library operations are preferred over Python loops for the same reason:
the library is both the smaller change and the faster one.

## 4. Dependency policy — external deps freely, bases covered

A dependency is allowed when it removes more code than it adds. It is never
invisible. Every direct dependency has a registry row and every library enters the
platform through an owned seam.

**Registry.** `configs/dependencies.yaml` — one row per direct dependency:

| field | meaning |
|---|---|
| `name` | package name (version spec stays in `pyproject.toml` — that is the version SSOT) |
| `owner` | role responsible for upgrades and removal |
| `scope` | `platform` \| `project` \| `dev` \| `optional` |
| `purpose` | the capability it provides |
| `adapter` | module that wraps it (empty only for dev tooling) |
| `license` | SPDX id, recorded for provenance only |
| `added` | date, so age is visible |

**Rules.**

- Hash-pinned through `uv.lock`; CI syncs with `--frozen`. No floating ranges at
  runtime.
- Licenses are **recorded, not gated**: this platform is internal and undistributed,
  so no allowlist blocks a dependency and no human reviews one. The SPDX id is
  captured automatically for provenance. Revisit only if the project is ever
  published, sold, or distributed as a product.
- A vulnerability scan runs at the fast tier; the severity threshold is config, and
  an advisory is never silently ignored.
- **One library per capability.** A second library that overlaps an existing one is
  a consolidation finding (`CONSOLIDATION.md`), not a preference.
- Import boundary: `src/broadway` imports `platform`-scoped deps only;
  `project`-scoped deps live under `project/`; heavy deps are optional extras and
  never on the core import path.
- Git/URL dependencies require an owner and an expiry date in the registry; no
  silent exceptions.
- **Adapter seam:** every library is reached through an owned module (Adapter /
  Facade). Swapping the library must not change a caller. Dev tooling is exempt.
- Removing or replacing a dependency is a new artifact, not an in-place edit: one
  commit, registry row updated, callers re-pointed.

## 5. Boundaries

- `src/broadway/` — data-agnostic platform machinery; no dataset terms, column
  names, thresholds, or project paths.
- `project/` — the dataset binding: config + thin glue only; no new platform logic.
- `project/experiments/` — analysis scratch; imports platform helpers, never
  reimplements them.
- `scripts/` — deterministic operations. A repeated manual sequence routes through a
  script named in `configs/tooling.yaml`.

## 6. Enforcement

- `ruff`, `mypy`, `vulture`, config-schema load — the existing fast tier.
- Dependency gate: every direct dependency resolves to a registry row; scope check;
  license recorded (not gated).
- Vulnerability scan at the fast tier once the scanner lands; severity threshold is
  config and an advisory is never silently ignored. Security is separate from
  licensing and is not relaxed by the internal-use policy.
- Import-boundary check: `src/` importing a `project`-scoped dep fails.
- Review: the pass-2 questions (§2) and one-library-per-capability (§4) are
  reviewer duties until the gates exist.
