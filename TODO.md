# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Project-style refactor** — waves 1-3 merged (k8s/optuna/mlflow finalized,
  auto-teardown + `verify_experiments.py` in). What remains from that stream:
  - `project/` dataset binding — partial: `project/data.py` (reads
    `configs/project/taxi.yaml`) owns `read_training_sample`,
    `load_stratified_sample`, zone lookup; multivariate `_setup.py` already
    imports zone constants from it. Still to move: univariate `_common.py` +
    mlflow `_common.py` loaders, and the multivariate importlib-load of
    univariate modules — one project-level binding.
  - Experiment knobs single source — partial: platform side done
    (`configs/{experiment,step,flow,dataset,environment,project}/` consumed by
    `broadway.config.loader.load_config` / `ProjectConfig`). Tutorial
    experiments still use module constants (univariate `_common.py`) or
    per-experiment `config.yaml` (multivariate).
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
- **Q-Q doc pass (remaining)** — README/dataflow already document
  `qq_zones`/`qq_markers`; only `src/broadway/stats/API.md` drift vs the
  zones/markers/raw-log work is left to check.

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
- **Pin sample/evidence, then refresh reports** — sample parquet is still
  gitignored/regenerable; `project/data.py` has seeded sampling but no pinned
  evidence commit.
- W1-flagged: Q-Q style constants Python-vs-YAML decision; hardcoded
  `figures/` path prefix in `src/broadway/discover/qq.py` (still
  `f"figures/{basename}"` despite `reports/paths.py::FIGURES_DIR`).

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
