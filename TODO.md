# TODO — current state / next / deferred

Ephemeral task tracker. `HANDOFF.md` is timeless; this file holds "what's next."
Update freely; remove entries when done (git history is the record).

## Current

- **Step 24 — influence plot (Cook's circular bubbles)** — script written
  (`experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py`),
  **not run, not committed** (run was aborted, cache was full). Finish:
  ```bash
  MPLCONFIGDIR=/home/opc/ONE/broad-way/.mplconfig .venv/bin/python experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py
  git add experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py
  git commit -m "feat(experiments): influence plot with Cook's distance bubbles (24)"
  git push origin taxi
  ```
  Plots leverage (x) × externally studentized residuals (y), bubble area ∝
  Cook's distance, 2p/n & 3p/n leverage lines, ±t residual cutoff, D > 4/n
  influential in red, top-5 trips annotated. Log-fare model, 42,806 trips.

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
- Scripts committed; PNGs/JSONs in gitignored `experiments/results/`;
  `diagnostics_experiment/` + `legend_experiment/` are gitignored references.
- Test gate: full `pytest` only when `src/`/`configs/`/`project/`/`tests/`
  change (suite currently has the 7 pre-existing CLI failures above).
- `uv` needs `UV_CACHE_DIR` inside the workspace (sandbox); matplotlib needs
  `MPLCONFIGDIR` (`.mplconfig/`).
