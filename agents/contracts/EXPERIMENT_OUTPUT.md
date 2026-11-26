# EXPERIMENT_OUTPUT.md — the output contract for experiment steps

Every artifact an experiment step writes is read by a human. A table dump or a
raw row sample does not answer "what did this step find?" — a figure does.
This contract is enforced by `scripts/check_experiment_outputs.py` (gate
`experiment-outputs`, part of the fast CI tier) and pinned by
`tests/test_experiment_outputs.py`.

## The rules

1. **Answer the step's own question.** The module docstring states the question
   (first line). The figure must answer it; decoration is a violation.
2. **Compare, don't just describe.** Where a reference exists (source data,
   baseline, prior step), the figure overlays it. A single raw distribution is
   not evidence of anything.
3. **Aggregates only, never raw rows.** No head/sample dumps, no per-row tables
   in a published figure or report. Shape, dtypes, missingness, quantiles,
   deltas — never datapoints.
4. **Deterministic.** Seeded sampling or none; a rerun must reproduce the
   figure byte-for-byte where the format allows.
5. **One figure per step; panels = the dimensions compared.** Tables are
   evidence under the figure, not the headline.
6. **Publish a decision record.** One metric, its 95% interval, the
   config-declared threshold and direction, and the **computed** verdict
   (`act` / `investigate` / `no_action`) plus the owner's action. Prose is not
   evidence: "the sample drifts" is a claim; "ks_fare_amount = 0.0120
   [0.0111, 0.0149] vs max 0.0100 at n=50,000 → act" is a decision.
   ```json
   {"question": "...", "metric": "ks_fare_amount", "value": 0.0120,
    "interval_low": 0.0111, "interval_high": 0.0149,
    "min_effect": 0.01, "null_quantile": 0.95,
    "threshold": 0.01, "null_threshold": 0.0077,
    "direction": "max", "verdict": "act",
    "action": "review the fare-policy exclusion before training",
    "owner": "fare_prediction", "sample_size": 50000}
   ```

   **The rule is sample-size calibrated.** An absolute threshold does not
   travel: a KS distance of 0.01 is noise on 1,000 rows and a real signal on
   8.5M. The effective threshold is `max(min_effect, null_quantile of the
   no-effect permutation null at this n)`; below `min_sample_size` the verdict
   is `insufficient` ("collect at least N rows") instead of a false act/ignore.
   The verdict is never hand-written: `Decision.evaluate` derives it from the
   interval (act when the whole interval clears the threshold; investigate
   when it straddles; no_action otherwise).

## Required outputs

- `<NN>_<name>.png` — the figure answering the question.
- `<NN>_<name>.md` — the aggregate report (the numbers behind the figure).
- Optional `<NN>_<name>_*.csv` — evidence tables the dashboard renders below
  the figure.

The dashboard opens on the figure and renders `<stem>*.csv` as evidence tables
(`src/broadway/reports/experiments_dashboard.py`).

`src/broadway/reports/decision.py` owns the record and the verdict rule;
`src/broadway/reports/step_output.py` is the writer. `StepOutput` requires a
`Decision`, refuses to write without a saved figure, and emits
`<stem>_decision.json` next to the report. The dashboard renders the record as
a coloured verdict badge on the step page.

## Where decisions must exist

The contract is not only for experiment scripts; every pipeline step that a
human reads must publish a decision record. This is the coverage matrix the
ratchet grows into (`PIPELINE_DECISION_ENFORCED`):

| Step | Decision to publish |
|---|---|
| `etl` | rows dropped / contract violations vs threshold → proceed or fix the source |
| `eda` | per-feature correlation and test (effect size + q-value) → keep/drop; outlier share → winsorize/act |
| `stats` | assumption tests → parametric vs non-parametric method |
| `features` | keep/drop/transform per feature (importance, effect, stability) |
| `baseline` | model vs naive baseline delta with CI vs threshold → is the model worth it |
| `training` | HPO best vs default with CI → adopt or keep default |
| `evaluate` | coverage vs nominal and promotion delta vs threshold → ship or hold |
| `tracking` | train-vs-serve drift (PSI/KS) vs threshold → retrain or monitor |
| `anomaly` | flagged share vs expected contamination → tune or act |

## The ratchet

`ENFORCED` in `scripts/check_experiment_outputs.py` lists the series that must
publish a question + figure (currently `fare_prediction`, `timeseries`).
`DECISION_ENFORCED` lists the individual steps that must build a `StepOutput`
with a decision record (currently `fare_prediction/01_generate_sample`). An
entry is added once it complies. **Never remove an entry.** The rule itself
lives in `project/config/experiments/fare_prediction.yaml` under
`decisions.<step>` — one threshold, one direction, one action, one owner.

Everything else is reported as an advisory backlog on every gate run, so the
gap is visible rather than forgotten.

## How to comply

- Write the question as the module docstring's first line.
- Build one figure and call `fig.savefig(<step>.png)`.
- Write aggregates only (describe/quantiles/deltas/funnel counts).
- Do not publish raw rows.

## Known advisory backlog (not enforced yet)

`mlflow` (1 step), `more_modeling` (19), `multivariate` (3),
`polynomial_regression_et_all` (1) — most miss the docstring question, several
also miss a figure. They join `ENFORCED` when fixed.
