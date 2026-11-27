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
    "threshold": 0.01, "null_threshold": 0.0077, "direction": "max",
    "verdict": "act", "action": "review the fare-policy exclusion before training",
    "owner": "fare_prediction", "sample_size": 50000,
    "evidence": {"baseline": "...", "segments": [{"segment": "...", "value": 0.0, "sample_size": 1}],
                 "sensitivity": {"half_sample": 0.0}, "falsifier": "...",
                 "provenance": {"artifact": "...", "seed": "42"}}}
   ```

   Every segment and sensitivity entry carries the n it was measured on; a
   segment measured on the whole frame while the record's `n` is a draw is the
   mixed-n defect the gate rejects. A sensitivity entry that equals the value
   at the same n is not a sensitivity.

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

## The evidence bundle (required)

A verdict without the evidence behind it is a claim. `Decision` therefore
requires an `Evidence` object, validated by construction:

| Field | What it must carry |
|---|---|
| `baseline` | what the metric is measured against (source, naive model, null) |
| `delta` | the difference from that baseline, when one exists |
| `segments` | at least one re-measurement inside a slice (weekday/weekend, borough, size band) — a slice that would expose a hiding failure |
| `sensitivity` | at least one robustness check (half sample, second seed, alternative knob) |
| `falsifier` | the result that would reverse the verdict |
| `provenance` | artifact + hash, source path, seed, exclusions |

A step that cannot supply one of these records why in the report instead of
omitting it silently.

## How to instruct an agent to surface more evidence

Put this in every build/delegation prompt (and quote this file):

> Every step you build must publish a `StepOutput` decision record: one metric
> with its 95% interval and n; the baseline it is measured against and the
> delta; at least one segment re-measurement; at least one sensitivity check;
> the falsifier; and provenance (artifact + hash, seed, exclusions). The
> verdict is computed by `Decision.evaluate` against a sample-size-calibrated
> threshold — never typed. If you cannot supply a piece of evidence, say so in
> the report; do not drop it silently. The gate (`experiment-outputs`) fails
> the build when the record or its evidence is missing.

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
publish a question + figure (currently empty: the Olist substrate publishes
through the pipeline steps, not a script series). `DECISION_ENFORCED` lists
the individual script pairs that must build a `StepOutput` with a decision
record (currently empty for the same reason). An entry is added once it
complies. **Never remove an entry merely for falling out of compliance**; an
entry is retired only when its dataset/series is removed (the predecessor
series were retired with the owner-approved dataset replacement). The rule itself lives in
the series' `project/config/experiments/<series>.yaml` under `decisions.<step>`
— one threshold, one direction, one action, one owner.

Everything else is reported as an advisory backlog on every gate run, so the
gap is visible rather than forgotten.

## How to comply

- Write the question as the module docstring's first line.
- Build one figure and call `fig.savefig(<step>.png)`.
- Write aggregates only (describe/quantiles/deltas/funnel counts).
- Do not publish raw rows.

## Known advisory backlog (not enforced yet)

The predecessor advisory backlog (`mlflow`, `more_modeling`, `multivariate`,
`polynomial_regression_et_all`) was retired with its dataset. The Olist
project enforces the `etl` and `eda` pipeline records; the classification
baseline/features/training/selection/HPO/evaluate chain is the next unit and
re-joins `PIPELINE_DECISION_ENFORCED` when it publishes its records.
