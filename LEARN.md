# Project Handoff — Logistics ML Learning Project (Session 2)

Continuation of the original pipeline-engineering + statistical-testing
handoff. That document covered `process.py` refactoring, data contracts,
and the ANOVA/Games-Howell borough analysis (scripts `01`–`07`). This
session picked up the **statistics track** and pivoted from "does borough
matter" (already answered: yes) to "is OLS actually a valid model here,"
before moving into feature engineering or a non-linear baseline.

Nothing in the pipeline-engineering track changed this session.

---

## 1. CLT / LLN — why the ANOVA assumption-checking was (mostly) justified

Before building on the ANOVA work, we checked whether all that
assumption-checking (Levene's, Shapiro, Welch's, Games-Howell,
Kruskal-Wallis) was actually necessary, or over-engineered.

- **LLN** only guarantees sample means converge to true means as n grows.
  It's why "mean duration per borough" is a meaningful quantity despite
  raw right-skew — it says nothing about test validity.
- **CLT** is what actually matters for ANOVA: it's the sampling
  *distribution of the group means* that needs to be normal, not the raw
  data. At Manhattan/Queens scale (n=155K / 15K) CLT does real work — the
  Shapiro p≈0 there is close to a non-finding, since Shapiro over-rejects
  at large n regardless of how normal the data practically is. At Bronx/
  Staten Island scale (n=319 / 84), skew + small n means CLT convergence
  is genuinely less certain.
- **CLT does not fix unequal variance.** Levene's test showed a ~7x
  variance ratio (Bronx std=25.6 vs Manhattan std=9.6) — this is a
  structural mismatch with standard ANOVA's pooled-variance assumption,
  and no amount of additional data averages it away.

**Conclusion:** Welch's ANOVA / Games-Howell were necessary — driven by
variance heterogeneity, not sample size. Kruskal-Wallis was good practice
but not strictly required given N, mainly useful as a cross-check for the
small groups. This reasoning turned out to directly predict the OLS
residual findings below.

---

## 2. OLS baseline + residual diagnostics

**Script:** `08_ols_residual_diagnostics.py`
**Model:** `trip_duration_minutes ~ trip_distance + C(pickup_borough)`
**Sample:** 199,974 rows, stratified by borough

### Results

- **R² = 0.642** — distance + borough alone explains a solid majority of
  variance.
- **Breusch-Pagan: p ≈ 0 → heteroskedastic.** Confirmed the prediction
  from Levene's test. Residual std by borough (6.3 Manhattan → 21.4 Bronx)
  tracks the original Games-Howell group-std ordering almost exactly —
  the borough dummy shifts the mean but does **not** absorb the variance
  difference.
- **Jarque-Bera: p ≈ 0, skew = 2.264, kurtosis = 25.491 → heavy
  right-tailed, non-normal residuals.** Magnitude matters more than the
  p-value here — kurtosis=25.5 indicates a small number of badly-underpredicted
  long trips dominate the tail, not mild non-normality.
- **Durbin-Watson = 1.994 — INVALID AS COMPUTED, caveat, not a finding.**
  This was computed on a stratified *random* sample; row order had no
  relationship to `pickup_datetime`. A DW≈2 on shuffled rows just proves
  "shuffled rows aren't autocorrelated with each other" by construction —
  it says nothing about real temporal autocorrelation. Flagged for
  redo (see script `10` below). **Do not cite the 08 DW value as evidence
  of anything.**

### Bottom line
Both formal heteroskedasticity and non-normality are confirmed, at a
magnitude that matters (not just significance-by-large-n). This is the
same variance-heterogeneity problem from the ANOVA work re-appearing in
regression residuals — expected, given the CLT/LLN reasoning above, but
now confirmed directly rather than inferred.

---

## 3. Follow-up scripts written this session (not yet run)

All four live in `learning/stats/`, continuing the numbering from the
original session's `01`–`07`.

| # | Script | Purpose | Status |
|---|---|---|---|
| 9 | `09_log_target_ols.py` | Does `log(duration) ~ ...` fix skew/kurtosis? Does it fix variance (predicted: no, per `06`'s group-level finding)? Also refits baseline with HC3 robust SEs — doesn't fix heteroskedasticity, just makes reported p-values/CIs valid despite it. | Written, not run |
| 10 | `10_durbin_watson_time_ordered.py` | Redo DW properly: sort by `pickup_datetime`, use a **contiguous time slice** (not random sample), plot ACF over a bounded lag window to catch cyclical (e.g. daily) autocorrelation that a single DW scalar would miss. | Written, not run. **`TIME_SLICE_START`/`END` are placeholders (`2024-01-01`–`2024-01-31`) — must be set to real coverage dates before running, or it silently loads 0 rows.** |
| 11 | `11_interaction_ols.py` | `trip_duration_minutes ~ trip_distance * C(pickup_borough)` — lets each borough have its own distance→duration slope, not just a shifted intercept. Includes nested F-test (`anova_lm`) vs. baseline, per-borough slope table, and re-run residual diagnostics (interaction should raise R² and may reduce skew, but heteroskedasticity is expected to persist — it fixes non-linearity, not variance heterogeneity). | Written, not run |
| 12 | `12_lgbm_baseline.py` | Non-linear baseline using full `ENGINEERED_FEATURES`. Uses a **time-based train/test split** (not random) since `10` establishes real temporal structure exists in the data — a random split would leak adjacent-in-time trips across train/test and overstate performance. Evaluates MAE/RMSE plus a **p90-tail MAE** specifically, since the whole motivation for this script was OLS's heavy-tail failure (kurtosis=25.5) — aggregate R²/MAE would hide whether that's actually fixed. | Written, not run. **Assumes `FeaturePipeline.fit()`/`.transform()` signature and that `TARGET` survives into the transformed frame — check against actual `features/pipeline.py` before running, since the original session's handoff notes `validate_engineered_schema` deliberately excludes the target (transform() is also used at inference time with no target present). May need to extract `y` before calling `.transform()`, not after.** |

### Known comparability gap
`08`/`09`/`11` (OLS) use a random stratified sample. `12` (LGBM) uses a
time-based split. **Do not compare R²/MAE across them directly yet** —
either score the OLS models on `12`'s time-based holdout, or note this
explicitly if reporting results. Flagged in `12`'s docstring but easy to
forget when eyeballing printed metrics side by side.

---

## 4. Updated methodology and order of operations

Each step below depends on the answer to the step before it, so the
order isn't arbitrary — running out of sequence risks building on an
unverified assumption (e.g., engineering time-based features before
confirming temporal autocorrelation is real).

### Step 1 — Fix and run `10` (time-ordered Durbin-Watson) first
This has to come first because it gates decisions in `12`:
- Fix the placeholder dates (`TIME_SLICE_START`/`END`) to real coverage
  dates — otherwise the script silently loads 0 rows and produces a
  false "no autocorrelation" result.
- Use a **contiguous time slice**, not a random sample — DW and ACF are
  meaningless on shuffled rows (this is exactly why `08`'s DW=1.994 was
  invalid, not a finding).
- Read the **ACF plot over a bounded lag window**, not just the scalar
  DW statistic — a single DW value can miss cyclical (daily/weekly)
  structure entirely.

**This determines:** whether `12`'s feature set needs explicit hour/day/
season features, or whether the time-based split already captures that
structure implicitly.

### Step 2 — Run `09` and `11` (cheap, confirmatory OLS follow-ups)
Independent of `10`'s outcome, so these can run in parallel or right after:
- **`09` (log-target OLS):** Tests whether `log(duration)` fixes skew/
  kurtosis. Prediction (from the ANOVA variance work) is that it will
  *not* fix heteroskedasticity — check this against the HC3 robust-SE
  refit, which doesn't cure heteroskedasticity but makes reported
  p-values/CIs valid despite it. Treat "did skew improve" and "are the
  inferential stats now trustworthy" as two separate questions.
- **`11` (interaction OLS):** Lets each borough have its own slope
  instead of a shifted intercept. Use the nested F-test (`anova_lm`)
  against baseline as the actual support for "interaction improves fit,"
  not just an eyeballed R² change. Expect R² up / skew possibly down,
  but heteroskedasticity to persist — interactions fix non-linearity,
  not variance heterogeneity. If heteroskedasticity does *not* persist,
  treat that as a surprising result worth re-checking rather than
  accepting at face value.

Both should re-run the same residual diagnostics as `08`
(Breusch-Pagan, Jarque-Bera) so results are directly comparable to the
baseline on identical tests.

### Step 3 — Verify the pipeline contract, then run `12` (LightGBM baseline)
Before running, confirm against actual `features/pipeline.py`:
- `FeaturePipeline.fit()`/`.transform()` signature matches what `12`
  assumes.
- Since `validate_engineered_schema` deliberately excludes the target
  (transform() is also used at inference time with no target present),
  `y` likely needs to be extracted **before** calling `.transform()`,
  not after — verify rather than assume.

Then run with:
- **Time-based train/test split** (already correctly specified in `12`)
  — justified directly by `10`'s findings, not just as best practice.
- **p90-tail MAE** as the key metric, not aggregate R²/MAE — the whole
  motivation for this script is OLS's tail failure (kurtosis=25.5), so
  aggregate metrics would hide whether that's actually fixed.

### Step 4 — Cross-model comparison, done correctly
Real methodological trap here: OLS models (`08`/`09`/`11`) are scored
on a random stratified sample; `12` uses a time-based split. **Do not
compare R²/MAE across them as-is.** Correct approach is one of:
- Re-score the OLS models on `12`'s time-based holdout, or
- Report both sets of scores side by side with an explicit note that
  they are not on comparable splits.

### Step 5 — Only after `12` has a working baseline
These are correctly deprioritized until there's a working non-linear
model to contextualize them against:
- **Correlation testing** (`trip_distance` vs. `trip_duration_minutes`)
  — run **both Pearson and Spearman**. Pearson assumes linearity, and
  the skew already established in this data makes Spearman the more
  trustworthy of the two; disagreement between them is itself
  informative about non-linearity.
- **Chi-square** on categorical features.
- **Leakage pass** on `ENGINEERED_FEATURES` — arguably should move
  earlier, since leakage would invalidate `12`'s results outright. Worth
  doing as a pre-check before trusting `12`'s p90 MAE, not strictly
  after.
- **ADF/stationarity test** — better motivated once `10`'s ACF shows
  whether real cyclical structure exists, since stationarity and
  autocorrelation are related diagnostics.

### Step 6 — A/B testing (later-stage)
Needs ≥2 trained models to compare. Per the autocorrelation question
under active investigation this session, **the split unit must be time
window, not individual trip** — splitting by trip would leak temporally
correlated observations across arms the same way a random train/test
split would leak them across `12`'s train/test.

### Open gap worth flagging
No true held-out final validation set has been reserved yet — every OLS
variant and the LightGBM baseline are being iterated against the same
time-based holdout from `12`. Before reporting a final performance
number, consider carving out a separate slice now, before more models
get tuned against that holdout, or the p90 MAE comparisons in Step 4
risk being optimistic.

---

## 5. Suggested next steps (original, superseded by Section 4 ordering above)

1. **Run `10` first**, before anything else — it needs correct date-range
   placeholders filled in, and its output (whether there's a real daily/
   weekly autocorrelation cycle) should inform whether `12`'s feature set
   needs explicit hour/day/season features or whether LightGBM's split
   structure already captures enough of it implicitly.
2. Run `09` and `11` — cheap, confirmatory, mostly settle the "is OLS
   salvageable" question one way or the other.
3. Run `12`, after checking the `FeaturePipeline` signature assumption
   above. Compare tail performance (p90 MAE) against OLS's known weak
   point specifically, not just aggregate R²/MAE.
4. Once `12` has a working baseline: revisit the original session's
   flagged next steps — correlation testing (`trip_distance` vs.
   `trip_duration_minutes`, Pearson vs. Spearman), chi-square on
   categorical features, leakage pass on `ENGINEERED_FEATURES`, and the
   ADF/stationarity timeseries analysis (now more clearly motivated if
   `10`'s ACF plot shows real cyclical structure).
5. A/B testing is still a later-stage item — needs ≥2 trained models to
   compare, and per the earlier discussion, the split unit should be
   **time window, not individual trip**, given the autocorrelation
   question this session is actively investigating.