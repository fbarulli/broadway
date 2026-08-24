# TODO — current state / next / deferred

Ephemeral task tracker. `agents/ledger/HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Statistical testing** — next work stream (user-directed, 2026-08-16).
  Scope to be set by the user: platform `src/broadway/stats` testing vs.
  hypothesis tests on the taxi data vs. tutorial continuation. Do not start
  until the specific direction is given.

Standing rules live in `agents/ledger/HANDOFF.md` / `agents/contracts/WORKER_CONTRACT.md` (config
files only, no env, no hardcoded values, data-agnostic `src/`, tests with
every src change, fail loud, one logical change per commit).

## Next

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
- **Refresh reports from pinned evidence** — `ratecode1_sample.parquet` +
  `ratecode1_sample.json` are pinned; reports that embed their numbers can now
  be refreshed reproducibly.
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
- Series state: steps 01-34 committed (12 merged into 11, 30 merged into 29).
  Step 29 =
  estimate size + 2x2 plots; 31/32 = NYC-style time buckets; 33 = metered-cost
  + forensics (official 2024 TLC `extra` is dirty — clean rows show a $1.00
  overnight surcharge, no peak); 34 = OLS vs LightGBM (OLS wins on 2 features).
- `experiments/mlflow/` — MLflow model battle (`01`), explainability (`02`),
  k8s optuna worker (`03`). Runs in `mlruns/` (gitignored); metrics CSVs
  tracked. `pyproject.toml` gained shap, lime, optuna-integration, psycopg2.
- CSVs are named after the producing step (28-34 + mlflow battle).
- Scripts committed; PNGs in gitignored `experiments/results/` (only
  `ratecode1_sample.parquet` + `ratecode1_sample.json` are pinned);
  `diagnostics_experiment/` + `legend_experiment/` are gitignored references.
- Test gate: full `pytest` only when `src/`/`configs/`/`project/`/`tests/`
  change (currently 516 passed, 0 failed).
- `uv` needs `UV_CACHE_DIR` inside the workspace (sandbox); matplotlib needs
  `MPLCONFIGDIR` (`.mplconfig/`).
