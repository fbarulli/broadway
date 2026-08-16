# DOING.md — what's left

Live task tracker (ephemeral; HANDOFF.md/AGENT_CONTRACT.md are the timeless
docs). Everything below `ad87b1e` on `taxi` is committed + pushed to
`github.com/fbarulli/broadway`.

## In flight

1. **Step 24 — influence plot (Cook's circular bubbles)**
   - Script written: `experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py`
   - **Not yet run, not committed.** The run was aborted (cache full).
   - To finish:
     ```bash
     MPLCONFIGDIR=/home/opc/ONE/broad-way/.mplconfig .venv/bin/python experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py
     git add experiments/univariate/fare_amount_trip_distance/24_ratecode1_influence_plot.py
     git commit -m "feat(experiments): influence plot with Cook's distance bubbles (24)"
     git push origin taxi
     ```
   - What it does: leverage (x) × externally studentized residuals (y),
     bubble area ∝ Cook's distance, reference boundaries at 2p/n & 3p/n
     leverage, ±t residual cutoff, D > 4/n influential in red, top-5 trips
     annotated with miles. Same log-fare model as 23 (42,806 trips).

## Open decisions (ask user)

2. **Step 23 standardization** — Cook's distance index plot was left
   single-panel ("single plot"). User option: add the bottom stats legend
   (BP/JB/skew/kurtosis) like 04/13/14/15/19, or keep as-is.
3. **Merge steps 11 + 12** into one file (reuse directive) — proposed, not
   approved.
4. **7 pre-existing CLI test failures** (`tests/test_cli.py`, e.g.
   `test_train_without_dataset_still_dispatches` exits 2) — verified
   pre-existing on clean `ca4e7ff`, investigation offered, not taken.

## Notes / conventions to remember

- Working dataset: `ratecode1_sample` (42,806 metered trips after
  fare > 2.50, duration > 0 & < 240 min) — `WORKING_DATASET` + `load_working` /
  `load_metered` in `_common.py`.
- All metrics persist to `ratecode1_sample.json` (gitignored, regenerable via
  `17_ratecode1_metrics.py` + step scripts).
- Results PNGs/JSONs live in gitignored `experiments/results/`; scripts are
  committed. `diagnostics_experiment/` + `legend_experiment/` are gitignored
  reference images.
- Test gate: full `pytest` only when `src/`/`configs/`/`project/`/`tests/`
  change (experiment-only changes skip it; suite currently has 7 pre-existing
  CLI failures).
- `uv` needs `UV_CACHE_DIR` inside the workspace (sandbox); matplotlib needs
  `MPLCONFIGDIR` (`.mplconfig/`).
