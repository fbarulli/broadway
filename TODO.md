# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Project-style refactor** — waves 1-3 merged (k8s/optuna/mlflow finalized,
  auto-teardown + `verify_experiments.py` in). What remains from that stream:
  - `project/` dataset binding — DONE (2026-08-16): `project/working.py` is
    the single working-dataset binding (path, filters, loaders, `time_bucket`,
    all from `configs/experiments/working.yaml`). Univariate `_common.py` +
    `_ols_bp.py` are thin re-exports; multivariate `_setup.py` imports it
    directly (importlib hack deleted); mlflow `_common.py` uses it too.
  - Experiment knobs single source — DONE: `configs/experiments/` now holds
    `working.yaml` (dataset), `multivariate.yaml` (moved from
    `experiments/multivariate/config.yaml`), `mlflow.yaml` (sample/split/seed/
    features). mlflow model recipes stay in code (they reference the platform
    model registry / sklearn classes).
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

- **Q-Q doc pass (remaining)** — README/dataflow already document
  `qq_zones`/`qq_markers`; only `src/broadway/stats/API.md` drift vs the
  zones/markers/raw-log work is left to check.

## Done (2026-08-16)

- **Step 23 standardization** — bottom stats legend (BP/JB/skew/kurtosis)
  added via the shared `attach_stats_legend`, matching 04/13/14/15/19.
- **Merge steps 11 + 12** — `11_ratecode1_dataset.py` now builds the dataset
  and renders the density scatter (former step 12 deleted; plot output is
  `11_ratecode1_scatter.png`).
- **Pin sample/evidence** — `ratecode1_sample.parquet` + `ratecode1_sample.json`
  are committed (gitignore exceptions added) so the 42,806-row working dataset
  and its metrics JSON are reproducible without raw data.

## Deferred

- **Cleanup/polish slice** — doc drift, dead code, silent posthoc skip,
  unlabeled Q-Q truncation. (Legacy results-index is NOT orphaned anymore:
  `reports/index.py`/`sequence.py`/`registry.py` are live + tested;
  `reports/audit.py` is taxi-free.)
- **Lineage-viz** (`graph_todo.md`) — `src/broadway/lineage/` exists
  (graph/mermaid/records/state/sample), but the learnGitBranching-style
  commit-tree renderer is still not built (`mermaid.py` emits plain
  `flowchart LR`).
- **Suggestion templates into config** — effect-size wording already moved
  (`configs/step/causal.yaml`, `configs/flow/stats_sequence.yaml`); the
  headlines/rationale in `src/broadway/timeline/suggest.py` are still
  hardcoded Python.
- **Refresh reports from pinned evidence** — sample + evidence JSON are pinned
  (see Done); reports that embed their numbers can now be refreshed
  reproducibly.
- W1-flagged: Q-Q style constants Python-vs-YAML decision; hardcoded
  `figures/` path prefix in `src/broadway/discover/qq.py` (still
  `f"figures/{basename}"` despite `reports/paths.py::FIGURES_DIR`).

## Experiment notes (univariate/fare_amount_trip_distance)

- Working dataset: `ratecode1_sample` (42,806 metered trips after
  fare > 2.50, trip_duration > 0, duration < 240 min) — binding owned by
  `project/working.py` (`configs/experiments/working.yaml`); `_common.py`
  re-exports it.
- All metrics persist to `ratecode1_sample.json` (PINNED — committed; also
  regenerable via `17_ratecode1_metrics.py` + the step scripts).
- Series state: steps 24-29 + 31-34 committed (30 merged into 29; 12 merged
  into 11). Step 29 =
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
