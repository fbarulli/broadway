# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Project-style refactor (DataScience paused 2026-08-16)** — audit of runs
  1+2: promote generic machinery from experiments to `src/`, thin experiments
  to config+src calls, single project-level dataset binding. Dependency
  breakdown (waves):
  - **Wave 1 — PARALLEL** (disjoint files; one commit each; test gate green
    per commit; pushes sequenced to avoid git races):
    - extend `evaluate/metrics.py`: full regression suite + binarized ROC/PR
      AUC (+ tests)
    - extend `data/splitter.py`: chronological split + stratified sampler
      (+ tests)
    - extend `training/mlflow_utils.py`: metadata logging (train/predict time,
      model size, no-artifact), `dataset_id` + `log_input` linking
    - extend `training/optuna.py` (RDBStorage); new `training/optuna_worker.py`
      (URL-from-config, retry, smoke test)
    - extend `stats/`: winsorize, modified_zscore, outlier_mask,
      estimation_table/standardized_coefs/scenario_dollars, residual-diagnostic
      + coefficient-forest plots (viz)
    - new `evaluate/explain.py`: SHAP / LIME / permutation / PDP-ICE /
      residuals
    - new `evaluate/feature_selection.py`: RFE curve
    - new `utils.py`: `require_keys` / `require_finite` validators
    - `project/` dataset binding: ONE loader for all experiments (kills the
      mlflow env override + multivariate importlib hack; three loaders -> one)
  - **Wave 2 — PARALLEL (after Wave 1 merged):** thin each experiment to
    config+src calls (univariate steps, multivariate 01-06, mlflow 01-03);
    `configs/experiments/*.yaml` as single source
    (seed/sample/split/features/models).
  - **Wave 3 — SEQUENTIAL:** parked k8s/optuna/mlflow thread (uses
    optuna_worker + mlflow_utils from src; config-as-files, no env).
  - Sequential gates: Wave 2 requires Wave 1 merged; Wave 3 requires the
    optuna/mlflow work; suite green on every commit.
  - **Coding style (mandatory for every refactor commit — AGENT_WORKER_CONTRACT
    + HANDOFF + user rules):**
    - No hardcoded values; YAML single source of truth (no `get(key, default)`);
      config files only, zero env vars (kill the `RATECODE1_PARQUET` override).
    - Data-agnostic `src/` (no column names / taxi terms — HANDOFF tiebreaker);
      dataset binding lives in `project/`.
    - Config remains the single source of truth EVERYWHERE — only
      `src/broadway/stats` functions take thresholds as parameters (no
      config/env/I/O reads *inside* stats); callers pass config values in.
    - Dispatch mechanics (AGENT_CONTRACT §3a): each Wave-1 task = ONE module;
      orchestrator re-reads the 1-3 target files immediately before dispatch;
      the contract enumerates the exact edit list (current content →
      replacement) + regenerated artifacts; short numbered steps; references
      AGENT_WORKER_CONTRACT.md; worker commits + pushes its own disjoint
      files; suite green per commit.
    - Type hints on all public functions; ~25-line single-responsibility
      functions; no dead/noise code.
    - Strategic logging only; catch only recoverable exceptions (fail loud).
    - Typed evidence + dumb renderers (renderers never compute); single-owner
      surfaces.
    - Tests with every src change (synthetic data, no taxi coupling in
      platform tests); suite green per commit; one logical change per commit;
      docs updated in the same commit.

- **Statistical testing** — next work stream (user-directed, 2026-08-16).
  Scope to be set by the user: platform `src/broadway/stats` testing vs.
  hypothesis tests on the taxi data vs. tutorial continuation. Do not start
  until the specific direction is given.

## Completed (Wave 3, 2026-08-16)

- **K8s + Optuna + MLflow finalization** — 3 optuna worker Jobs (one per model)
  completed on the kind cluster; studies + trials in the optuna DB, mlflow runs
  (params/metrics/dataset lineage) in the mlflow DB; endpoint logging on every
  pod. Root cause of the original saga: **optuna and mlflow shared one postgres
  database and clashed over Alembic's `alembic_version` table** — fixed by
  separate `optuna`/`mlflow` databases (init script creates the second). Other
  fixes: `optuna-init` Job creates studies once; config/secret mounted as
  FILES (zero env vars); workers use `broadway.training.optuna_worker` +
  `mlflow_utils` (Wave-1 src); base image + separate mlflow/worker images;
  mlflow 3.15 everywhere; Job completion semantics (no infinite restarts);
  postgres + mlflow PVCs (durable); `k8s/optuna/` manifests committed.
  Standing rules still apply: config files only (no env), single source of
  truth, data-agnostic, fail-loud, no drift-type additions without asking.

- **Auto-teardown + fast verification entry point** (2026-08-16, `cc99dab`):
  finished Jobs self-delete via `ttlSecondsAfterFinished: 600` on all four
  `k8s/optuna/` Job manifests (verified live: completed jobs + pods gone from
  the cluster ~10 min after finish); `k8s/optuna/teardown.sh` removes the
  whole stack and the kind cluster on demand;
  `verify_experiments.py` at the repo root is a ~seconds entry point that
  syntax-compiles every experiment script, validates all YAML configs through
  `require_keys`, imports the shared experiment modules, and spot-checks
  structural invariants (no training, no figures, no cluster).

## Next

- **Step 23 standardization** — Cook's index plot is single-panel; decide:
  add the bottom stats legend (BP/JB/skew/kurtosis) like 04/13/14/15/19, or
  keep as-is.
- **Merge steps 11 + 12** into one file (reuse directive).
- **7 pre-existing CLI test failures** (`tests/test_cli.py`, e.g.
  `test_train_without_dataset_still_dispatches` exits 2) — verified
  pre-existing on clean `ca4e7ff`; investigate when time allows.
- **Q-Q doc pass** — `README.md`/`dataflow.md` Q-Q sections predate the
  zones/markers/raw-log work; `stats/API.md` drift; raw/log comparison +
  marker legend not yet documented.

## Deferred

- **W2 main-sync** — taxi-free `main`; convert/exclude taxi-referencing tests (`test_contracts.py` real-parquet load, hardcoded taxi stats in `test_walkthrough.py`/`test_results.py`).
- **Cleanup/polish slice** — orphaned legacy results-index (`render_index`/`load_stats_sequence`/`RESULT_RENDERERS`), doc drift, taxi strings in `audit.py`, dead code, silent posthoc skip, unlabeled Q-Q truncation.
- **LoadAudit / ParsingPolicy**.
- **Lineage-viz** (`graph_todo.md`).
- Move suggestion templates + effect-size wording into config; delete legacy `report` wrapper.
- Pin sample/evidence, then refresh reports.
- W1-flagged: Q-Q style constants Python-vs-YAML decision; hardcoded `figures/` path prefix.

## Experiment notes (univariate/fare_amount_trip_distance)

- Working dataset: `ratecode1_sample` (42,806 metered trips after
  fare > 2.50, trip_duration > 0, duration < 240 min) — `WORKING_DATASET`,
  `load_working`, `load_metered` in `_common.py`.
- All metrics persist to `ratecode1_sample.json` (gitignored; regenerable via
  `17_ratecode1_metrics.py` + the step scripts).
- Series state: steps 24-29 + 31-34 committed (30 merged into 29). Step 29 =
  estimate size + 2x2 plots; 31/32 = NYC-style time buckets; 33 = metered-cost
  + forensics (official 2024 TLC `extra` is dirty — clean rows show a $1.00
  overnight surcharge, no peak); 34 = OLS vs LightGBM (OLS wins on 2 features).
- `experiments/mlflow/` — MLflow model battle (`01`), explainability (`02`),
  k8s optuna worker (`03`). Runs in `mlruns/` (gitignored); metrics CSVs
  tracked. `pyproject.toml` gained shap, lime, optuna-integration, psycopg2.
- CSVs are named after the producing step (28-34 + mlflow battle).
- Scripts committed; PNGs/JSONs in gitignored `experiments/results/`;
  `diagnostics_experiment/` + `legend_experiment/` are gitignored references.
- Test gate: full `pytest` only when `src/`/`configs/`/`project/`/`tests/`
  change (suite currently has the 7 pre-existing CLI failures above).
- `uv` needs `UV_CACHE_DIR` inside the workspace (sandbox); matplotlib needs
  `MPLCONFIGDIR` (`.mplconfig/`).
