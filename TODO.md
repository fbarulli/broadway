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

- **Statistical testing** — next work stream (user-directed, 2026-08-16).
  Scope to be set by the user: platform `src/broadway/stats` testing vs.
  hypothesis tests on the taxi data vs. tutorial continuation. Do not start
  until the specific direction is given.

## Parked (paused by user, 2026-08-16)

- **K8s + Optuna + MLflow finalization** — kind cluster `broadway` still
  running; postgres + mlflow healthy (migration ordering matters: mlflow must
  migrate first in isolation, then workers). Optuna workers hit the concurrent
  schema-creation race (torn `version_table`). Remaining gaps:
  - Deterministic race fix: `optuna-init` Job creates the studies once;
    workers only `load_if_exists`.
  - Verify worker → mlflow logging end to end; fix Deployment restart
    semantics (finished workers must not re-run); bake migration ordering into
    the manifests.
  - Config-files-only refactor: ConfigMaps/Secret mounted as FILES, zero env
    vars (also removes the `RATECODE1_PARQUET` env override).
  - Promote battle machinery to `src/` (data-agnostic): extend
    `evaluate/metrics.py`, `training/mlflow_utils.py` (metadata logging, no
    artifact), `training/optuna.py` (RDB support); new `optuna_worker.py`,
    `evaluate/explain.py`, `evaluate/feature_selection.py`, pipeline factory.
  - `configs/experiments/ratecode1_battle.yaml` as single source
    (seed/sample/split/features/models/experiment/tracking).
  - Durable postgres (PVC), mlflow artifact volume, non-demo Secret,
    mlflow 3.15.1 vs platform pin 2.22.1 (`docker/mlflow/Dockerfile`),
    commit `k8s/optuna/` manifests (currently uncommitted).
  - Hard rules from user: config files only (no env vars, 100% of the time),
    single source of truth, data-agnostic, no drift-type additions without
    asking first.
  - Infra requirements (user, 2026-08-16): create a shared **base image**;
    **pin mlflow to 3.15 wherever applicable** (incl. `docker/mlflow/Dockerfile`,
    currently 2.22.1); **separate images** (base / mlflow-server / optuna-worker
    — no monolithic image); **never allow pods to restart indefinitely**
    (completion semantics / Job-style, no infinite CrashLoopBackOff); **proper
    logging always** — inside containers and pods, log the hostname/IP/URL/DB
    endpoint actually in use at startup so it is verifiable which endpoint the
    process connected to.

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
