# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Project-style refactor** — waves 1-3 merged (k8s/optuna/mlflow finalized,
  auto-teardown + `verify_experiments.py` in). What remains from that stream:
  - `project/` single-loader dataset binding: one loader for all experiments
    (univariate `_common.py`, multivariate `_setup.py`, mlflow `_common.py`
    still each carry loader logic — move to one project-level binding).
  - `configs/experiments/*.yaml` as single source for experiment knobs
    (currently module constants / per-experiment `config.yaml`).
  - Coding style (mandatory for every refactor commit — AGENT_WORKER_CONTRACT
    + HANDOFF + user rules):
    - No hardcoded values; YAML single source of truth (no `get(key, default)`);
      config files only, zero env vars.
    - Data-agnostic `src/` (no column names / taxi terms — HANDOFF tiebreaker);
      dataset binding lives in `project/`.
    - Config is the single source of truth EVERYWHERE — only
      `src/broadway/stats` functions take thresholds as parameters (no
      config/env/I/O reads *inside* stats); callers pass config values in.
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

## Next

- **Step 23 standardization** — Cook's index plot is single-panel; decide:
  add the bottom stats legend (BP/JB/skew/kurtosis) like 04/13/14/15/19, or
  keep as-is.
- **Merge steps 11 + 12** into one file (reuse directive).
- **7 pre-existing CLI test failures** (`tests/test_cli.py`, e.g.
  `test_train_without_dataset_still_dispatches` exits 2) — do NOT reproduce in
  this env (suite is 516 passed / 0 failed); investigate only if they appear
  elsewhere.
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
  change (currently 516 passed, 0 failed).
- `uv` needs `UV_CACHE_DIR` inside the workspace (sandbox); matplotlib needs
  `MPLCONFIGDIR` (`.mplconfig/`).
