# Amazon consumer-behaviour dataset — stage 1 proposal (for adversarial review)

Status: **implemented, NOT approved.** This file is the review artifact required by
`agents/contracts/ADVERSARIAL_REVIEW.md`. A separate read-only reviewer must return
PROCEED / REVISE / REJECT; the commit messages carry `Adversarial: pending` until then.

Owner directive: replace the NYC-taxi dataset with
`swathiunnikrishnan/amazon-consumer-behaviour-dataset`, staged. **Stage 1 is this
change**: add the Amazon dataset end-to-end and keep the existing taxi configs,
experiments, and tests green. No taxi file, config, or data is removed in this stage.
The removal plan is §12 (write-only).

---

## 1. Target decision (SSOT for this stage)

| Field | Value |
|---|---|
| target | `Shopping_Satisfaction` |
| task | `regression` (ordinal 1–5 treated as numeric) |
| datetime column | none (`Timestamp` is `role: ignore`, per the directive) |
| split | `random`, `validation_size: 0.2` |
| target metric | `rmse` |
| model | `linear` (declared, not trained in this stage) |
| random_state | `42` (`configs/onboard.yaml:6`) |

**Soundness flag (do not read this as a silent switch).** The target is an ordinal
1–5 rating with a skewed, bounded distribution — `{1: 115, 2: 194, 3: 209, 4: 67,
5: 17}` at n=602. Regression makes the platform's RMSE/MAE + interval decision
machinery applicable, and the EDA/test machinery accepts it (`ColumnKinds.target`
with `TaskType.REGRESSION` → numeric). What this decision does **not** license:

- a claim that satisfaction is well modelled by linear regression — residuals will be
  bounded and heteroscedastic, and the top class has 17 rows;
- class-level accuracy or per-rating conclusions (only 17 rows rate 5);
- interval/coverage claims from `calibration` — no model is trained this stage.

The statistically proper model, if a model is ever built, is ordinal logistic (or
ordinal-aware LGBM), not OLS. The survey question this stage actually answers is an
association question (Spearman rank correlation), where the 1–5 coding is sound.
The directive is followed as written; the limitation is recorded here, not hidden.

## 2. What changed (file:line)

### 2.1 Platform — one read-time normalization, two dataset namespaces

| File | Change | Replaces |
|---|---|---|
| `src/broadway/config/schema.py:73-130` | `DatasetContract` gains `column_renames: dict[str,str] = {}`, `processed_subdir: str \| None`, `artifacts_subdir: str \| None`; validators reject an ambiguous rename map (unbound/target-collision/duplicate targets) and an absolute or `..` subdir at parse | nothing — the contract previously had no rename/normalize or namespace field, and `load_with_audit` read `dataset.path` raw |
| `src/broadway/data/normalize.py:17-63` (new) | `ColumnNormalizer.from_contract(contract).apply(frame)` — the one applier; identity when no renames; `source_name()` maps a contract column back to the file's label for pre-read projections; a declared source label missing from the frame raises `KeyError` (loud) | the duplicated raw reads in `loader` and `discover` |
| `src/broadway/data/loader.py:162` | `load_with_audit` applies the normalizer to the raw read **before** lookups and before every consumer validates | `READERS[ext](path)` raw |
| `src/broadway/data/loader.py:198-213` | `read_sample` translates a `columns=` projection through `source_name` before the polars `select`, then applies the normalizer | raw-label projection (raised `ColumnNotFoundError` for a contract name) |
| `src/broadway/discover/module.py:116-165` | `profile()` reads the contract through `config_path(...)` (overlay-aware) and normalizes the frame; profile/qq evidence and figures resolve through `ArtifactStore` (dataset namespace when declared); `discover --artifacts-subdir` records the namespace on a bootstrap contract | `CONFIGS_DIR / DATASET_DIR` read + raw frame + shared `reports/figures` |
| `src/broadway/data/loader.py:113-171` | `ProcessedStore` (processed data: canonical/split/feature files) and `ArtifactStore` (machine evidence: discover/tracking/evaluation): a dataset's `processed_subdir`/`artifacts_subdir` overrides the shared directory; `canonical_path` routes through `ProcessedStore` | the 10 duplicated `Path(cfg.environment.data_dir) / cfg.environment.processed_subdir` expressions (listed in §9) and the dataset-blind artifact roots |

Why `column_renames` and **not** `strip_column_names`: the file has
`Personalized_Recommendation_Frequency` (object, Yes/No/Sometimes) **and**
`Personalized_Recommendation_Frequency ` (int 1–5) — proven different variables: 0
elementwise-equal cells. Stripping the trailing space would collide those two into
one label and force a silent drop; an explicit rename keeps both with distinct
names. Adding an unused `strip_column_names` knob would be an option that buys
nothing (ER contract §5, rung 1). The map is config, not code: the contract names
both source labels and both target labels, and pydantic rejects an ambiguous map
at the boundary (`schema.py:95-118`).

Why `ProcessedStore` and **not** the suggested project step overlay: a
`project/config/step/etl.yaml` overlay is applied to **every** project step load
(`loader._load_yaml` deep-merges the project step file over the base), so renaming
`train_file`/`val_file` there would silently make a future **taxi** `etl` run write
Amazon-named artifacts — a capability regression for taxi, which the ER contract
forbids. A dataset-level processed namespace fixes both directions (Amazon cannot
overwrite taxi, taxi cannot overwrite Amazon) with one owner and no new naming
convention. Verified below that taxi's resolved paths are byte-identical to HEAD.

### 2.2 Small platform fixes folded in (same boundary classes)

| File | Change | Why |
|---|---|---|
| `src/broadway/eda/artifacts.py:69-84` | `EdaArtifactWriter.remove_figure(name, fmt)` deletes a same-stem figure the current run did not draw; `eda/module.py:176-186` calls it when a decision figure is `None` | a skipped panel (no numeric column pair) left the previous dataset's heatmap in place, linked by the dashboard beside the new record |
| `src/broadway/eda/module.py:206-227` | provenance `sample_size` is `min(config budget, source rows)`; the summary record gains the chosen feature's `effect_name` as its `unit` and the correlation status in provenance | the record claimed `sample_size: 50000` beside `sampled_rows: 602`, published a unitless statistic, and did not say why the correlation panel was absent |
| `src/broadway/eda/tests.py:138-150` | the ε² docstring pins the published definition: midranks in H, **no** tie correction in the denominator (tie-corrected is larger: 0.2515 vs 0.2299 on this family) | a cross-study reader must not mix the two corrections |
| `src/broadway/onboard/module.py:200-215` | the generated plan imports `DerivedFeature` only when it has derived features | the generated Amazon plan has none, and the unconditional import is an F401 on every generated plan without derived features |
| `tests/test_profile.py:81`, `tests/test_qq.py:638` | the two `profile()` tests point `BROADWAY_CONFIG_OVERLAY_DIR` at their isolated tree instead of patching `loader.CONFIGS_DIR` | the contract resolver moved to `config_path`; patching `CONFIGS_DIR` also redirects `viz.yaml` and breaks plotting |

### 2.3 The dataset binding (new, under `project/`)

| File | Content |
|---|---|
| `project/config/dataset/amazon.yaml` | 23 columns (post-rename names), `target`, `task`, `datetime_column: null`, `column_renames`, `processed_subdir: processed/amazon`, `artifacts_subdir: amazon` |
| `project/config/analysis/amazon.yaml` | init's analysis contract: goal, row definition, decision moment, success criterion, leakage note (cross-sectional ⇒ associations are not causal) |
| `project/config/experiment/amazon.yaml` | `loader: canonical`, `feature_plan: project.features.amazon:plan`, linear model, random 0.2 split, seed 42, `rmse` |
| `project/features/amazon.py` | generated by `init`, then hand-tuned: the three 1–5 rating columns (and `age`) are numeric `source` keeps rather than frequency-encoded counts; nominal strings keep their frequency encodings. 152 lines; imported by the loader when the ETL config binds the experiment |
| `project/config/experiments/amazon_survey.yaml` | the series' seed/iterations, `feature`, the 12-name `family` (the EDA ranking), and the `01_profile` decision rule (metric/direction/min_effect/min_sample_size/action/owner) |
| `project/experiments/amazon_survey/01_profile.py` | the ONE survey step (§7) |
| `project/experiments/results/amazon_survey/01_profile.{md,png}` + `_decision.json` + `_ranked.csv` + `_segments.csv` | committed output contract |
| `project/experiments/results/pipeline/eda/*` | **replaced** taxi EDA evidence with the Amazon run (§6) |

### 2.4 Ratchet + maps (the anti-stale-doc rule)

| File | Change |
|---|---|
| `scripts/check_experiment_outputs.py:30,48,120-160` | `ENFORCED` gains `amazon_survey`; `DECISION_ENFORCED` gains `("amazon_survey","01_profile")`; a decision-enforced pair now needs a record that validates through the canonical `Decision` (the AST `StepOutput` call alone no longer passes) |
| `agents/contracts/EXPERIMENT_OUTPUT.md:119-126` | ratchet paragraph lists the current `ENFORCED` / `DECISION_ENFORCED` sets |
| `README.md:819-820` | records `amazon_survey/01_profile` as publishing a record |
| `project.md:82-84` | structure map gains `amazon_survey/` |
| `tests/test_loader.py:224-310` | three focused public-behavior tests (§10); `tests/test_experiment_outputs.py` pins the enforced count at 13 and proves a missing record fails |

## 3. Dataset profile (verified against the real artifact)

Command (`python` on the raw CSV; `data/` is gitignored):

```
$ python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/raw/amazon/amazon_consumer_behavior_survey.csv")
...
EOF
(602, 23)
dups 0
nulls: {'Product_Search_Method': 2}
target {1: 115, 2: 194, 3: 209, 4: 67, 5: 17}
ts unique 601
cols repr: ['Timestamp', 'age', 'Gender', 'Purchase_Frequency', 'Purchase_Categories',
 'Personalized_Recommendation_Frequency', 'Browsing_Frequency', 'Product_Search_Method',
 'Search_Result_Exploration', 'Customer_Reviews_Importance', 'Add_to_Cart_Browsing',
 'Cart_Completion_Frequency', 'Cart_Abandonment_Factors', 'Saveforlater_Frequency',
 'Review_Left', 'Review_Reliability', 'Review_Helpfulness',
 'Personalized_Recommendation_Frequency ', 'Recommendation_Helpfulness',
 'Rating_Accuracy ', 'Shopping_Satisfaction', 'Service_Appreciation', 'Improvement_Areas']
a ['Yes','No','Sometimes'] b [1,2,3,4,5]  equal elems 0
```

dtypes: 18 `object` + 5 `int64` (`age`, `Customer_Reviews_Importance`,
`Personalized_Recommendation_Frequency `, `Rating_Accuracy `, target). Every file size
claim from the directive was re-checked: 172,521 bytes; 602×23; 0 duplicate rows; 2
null cells; 601 unique timestamps.

Other properties the step had to design around (all re-verified):

- `Purchase_Categories` is semicolon-joined multi-select (29 distinct strings).
- The two trailing-space headers are **distinct variables** from their space-free
  namesakes, not duplicates to drop (0 elementwise equal).
- `Timestamp` is a locale-formatted string (`2023/06/04 1:28:19 PM GMT+5:30`), not
  ISO; it is ignored this stage (per the directive) and is not the datetime column.

## 4. Hand-fixes over the `init` scaffold (every one recorded)

`init`/`discover` write through `CONFIGS_DIR` (base `configs/`), not the project
overlay (`src/broadway/onboard/module.py:315-333`, `src/broadway/discover/module.py:110-115`).
The bootstrap therefore ran `init` on a **temporary renamed copy** of the raw CSV
(`data/raw/amazon/.init_normalized.csv`, created and deleted in this task; the raw
CSV was never modified), wrote `configs/{dataset,analysis,experiment}/amazon.yaml`,
and those three files plus `project/features/amazon.py` were moved/kept as project
files. Then:

1. **`path`** — `init` recorded the temp copy. Hand-fixed to
   `data/raw/amazon/amazon_consumer_behavior_survey.csv`.
2. **`datetime_column`** — `""` from `--split-column ""` (random split). Hand-fixed
   to `null`.
3. **`column_renames`** — added by hand: discovery cannot know that a raw label with
   a trailing space is a different variable from its space-free namesake.
4. **`processed_subdir`** — added by hand: artifact namespace is a policy decision,
   not inferable.
5. **Comment block** in the contract records all four (so the next reader does not
   re-derive them).

The column `dtype`/`null_count`/`role` entries are `init`'s own output and were not
edited. Re-running `discover` on the raw path would regenerate raw trailing-space
labels and lose the renames — the committed contract is the artifact of record.

## 5. ETL — verified output (real command, exit 0)

```
$ python -m project.cli etl --dataset amazon --experiment amazon
INFO ... saved canonical (602 rows) to data/processed/amazon/amazon_canonical.parquet
INFO ... saved train (481 rows) and val (121 rows)
INFO ... etl decision: no_action — unexplained drop share 0.0000 (ceiling 0.1000)
INFO ... saved structural clean result to data/processed/amazon/amazon_clean.json
INFO ... manifest: wrote amazon (602 rows) to artifacts/tracking/data_manifest.json
INFO ... step complete: etl        (exit 0)
```

Independent verification on the produced artifacts:

```
canonical (602, 23)  train (481, 23)  val (121, 23)  sum 602
target dist {1: 115, 2: 194, 3: 209, 4: 67, 5: 17}
null cells 2 {'Product_Search_Method': 2}
dtypes {object: 18, int64: 5}
trailing-space labels: []
cols 23
```

`data/processed/amazon/etl_decision.json` is a canonical `Decision` (validated with
`Decision.model_validate_json`): `rows_dropped_unexplained_share = 0.0
[0.0, 0.0], threshold 0.1, n=602, verdict no_action`.

**The 2 null rows are kept, not dropped.** `canonicalize` only drops duplicates and
null-target rows (`src/broadway/data/cleaner.py:17-32`); the pandera raw schema is
built from every contract column with `nullable=True` (`src/broadway/contracts/pandera.py:44-56`),
and `Product_Search_Method` is a categorical feature. Row accounting: 602 in → 602
out, 0 explained drops, 0 unexplained drops. No frequency-encoding NaN token is
needed because the ETL does not encode; the generated feature plan's frequency
encoder handles the 2 missing cells when features are ever fitted (stage 2+).
This choice is stated, not inferred: **keep the rows, keep the nulls**.

ETL re-run determinism: the three parquet files are byte-identical on a second run
(sha256 before/after), and the EDA decision JSON is byte-identical too.

## 6. EDA — verified output (real command, exit 0)

```
$ python -m project.cli eda --dataset amazon
INFO ... eda: reading train split from data/processed/amazon/train.parquet (481 of 481 rows, stream 0)
INFO ... eda: reading val split from data/processed/amazon/val.parquet (121 of 121 rows, stream 1)
INFO ... eda: reading canonical frame from data/processed/amazon/amazon_canonical.parquet (602 of 602 rows, stream 2)
INFO ... eda: rows=602 sampled=602 missing_cells=2 constant=0 duplicates=0 outlier_columns=5
         features_tested=17 decisions={'act': 3, 'investigate': 7, 'no_action': 2, 'undecided': 5}
         compare=available
INFO ... eda: report written to project/experiments/results/pipeline/eda/eda.html
INFO ... step complete: eda       (exit 0)
```

Committed decision record (`project/experiments/results/pipeline/eda/eda_decision.json`,
validated by the canonical `Decision`):

```
metric = association_rating_accuracy_score
value = 0.22993  [0.18007, 0.29497]  95%
threshold = 0.05   null_threshold = 0.007443 (no-effect shuffle at n=602)
direction = max   verdict = act   owner = eda   sample_size = 602   min_sample_size = 100
segments = Female 0.1872 (n=352), Male 0.2325 (n=142), Others 0.3840 (n=19),
           Prefer not to say 0.3280 (n=89)
sensitivity = eta_squared 0.2703, iterations 200, alpha 0.05
provenance = dataset amazon; artifact train.parquet+val.parquet;
             source data/raw/amazon/amazon_consumer_behavior_survey.csv;
             sample_sha256 9858827b9b2fb5f4; sampled_rows 602; sample_size 602; seed 42
```

Threshold is sample-size calibrated: floor `min_effect = 0.05`
(`configs/step/eda.yaml:24`) raised by the measured permutation null at this n
(0.0074). The `insufficient` floor is 100 and n=602.

**The EDA evidence slot is replaced on purpose.** `project/config/step/eda.yaml:5`
points EDA at `project/experiments/results/pipeline/eda` — a dataset-agnostic slot.
The Amazon run now occupies it (decision + sidecar + 4 CSVs + 3 PNGs), replacing the
taxi evidence committed at HEAD. This is conscious and allowed by the directive
(taxi is being retired); the taxi tests stay green (§10). The rendered `eda.html`
stays local/ignored. The five skipped columns are recorded by the run
(`Timestamp`, `age`, `Purchase_Categories`, `Service_Appreciation`,
`Improvement_Areas` — sparse χ² / high-cardinality groups), not silently dropped.

## 7. Experiment series `amazon_survey` — one real survey question

Question (script docstring, first line): *which reported behaviour tracks shopping
satisfaction most strongly at n=602?*

- **Metric** — `|Spearman rho|` between the config-declared `feature`
  (`rating_accuracy_score`, raw `Rating_Accuracy `, ordinal 1–5) and the dataset's
  target (`cfg.dataset.target`). One measurement owner: `MeasurementEstimator` (iid
  bootstrap interval), turned into the record through `Measurement.to_decision`.
- **Selection-adjusted null (the rule's own null band)** — the feature was ranked
  first by EDA, so the null is the 95th percentile (`rule.null_quantile`) of **max
  |rho| over the WHOLE config-declared 12-feature family under ONE shared target
  permutation per draw** at n=602. The measurement is rewritten with
  `model_copy(update={"null_threshold": family_null})` before `to_decision`, and the
  step asserts the record's `null_threshold` equals the published
  `family_max_rho_null_q0.95_n602` sensitivity entry and is below `min_effect`
  (0.15). Published value: **0.116589**.
  The earlier draft published a 3-feature null (0.0948) and a per-feature shuffle
  (0.0771); the reviewer's independent probe reported ~0.1181. Re-deriving it here
  with the shared-permutation construction over the 12 declared features gives
  0.1166 with `scipy.stats.spearmanr` (midranks); 0.1176 if the feature codes are
  correlated without midrank tie correction; 0.1181 reproduces only under
  per-feature independent permutations with raw-code Pearson. The published number is
  the one the record's own statistic computes; all four variants are far below
  `min_effect` 0.15, so the verdict is insensitive to the choice.
- **Segments** — the shared `SegmentPlan.build` rule (`datetime_column` absent → the
  first low-cardinality column, here `Gender`), each level with its own n; `Others`
  n=19 is noisy and shown as such.
- **Sensitivity** — `rho_half_sample_n301 = 0.5123` (its own n, a genuine
  re-measurement, not a copy), the family null above, and `kendall_tau_abs_n602 =
  0.4502` as an alternative rank measure.
- **Provenance** — dataset `amazon`, artifact path + sha256 prefix, raw source, rows,
  seed, resampling, the family declaration, the segment rule, exclusions.

Verified run (`python project/experiments/amazon_survey/01_profile.py`, exit 0):

```
decision: .../01_profile_decision.json -> act
(satisfaction_rating_accuracy_rho 0.5015 [0.4275, 0.5729] vs 0.1500, null 0.1166, n=602)
```

The figure (committed PNG) answers the question in three panels: group means of
satisfaction by rating-accuracy level with n labels; the full-sample interval + half
sample + Gender slices against the rule and selection-null lines; and the ranked
12-feature family against the same lines. Aggregates only, no raw rows, seeded
(`seed: 42`, `iterations: 1000` in `project/config/experiments/amazon_survey.yaml:6-7`).
The script owns no column literal: target from the contract, feature/family from
config, segments from `SegmentPlan`, the CI multiplier from the measurement's
confidence, the null quantile from the rule.

Rerun note: `StepOutput` refuses to republish a record that is older than its script
when the content is identical (`StaleDecisionError`) — an edit that leaves the
numbers unchanged requires deleting the record first, then rerunning. That guard
fired once during this task and the record was regenerated from the current script.

No 602-row modelling pipeline was invented: the series has exactly one step.

## 8. Normalization proof (end to end)

`project/config/dataset/amazon.yaml` declares

```yaml
column_renames:
  "Personalized_Recommendation_Frequency ": recommendation_personalization_score
  "Rating_Accuracy ": rating_accuracy_score
```

`profile --dataset amazon` (real command) reports:

```
profile: wrote 23 columns to artifacts/discover/profile.json
amazon 602 23
nulls: {'Product_Search_Method': 2}
has trailing: []
renamed present: ['recommendation_personalization_score', 'rating_accuracy_score']
```

so the profiler (which indexes `contract.columns[col]`) sees exactly the contract's
labels, and the ETL's pandera validation (`build_raw_schema(dataset).validate(df)`)
accepts the normalized frame — `amazon_clean.json` records `columns_before` with both
clean names. The raw CSV bytes are untouched (gitignored; `sha256` of the raw file is
unchanged from the checkout).

## 9. Blast radius (every place the change reaches)

1. `DatasetContract` is constructed by: `discover._build_contract`, `onboard.build_dataset_contract`,
   `loader.load_config` (`config/schema.py` inside `PipelineConfig`), `project/data.py`,
   `project/features/__init__.py`, and many tests. New fields are optional with
   defaults; `_validate_column_renames` returns immediately on an empty map. No
   existing constructor sets them.
2. `ColumnNormalizer` is applied on three read paths (loader ×2, discover profile).
   With an empty map `apply` returns the same object — taxi and test datasets are
   untouched by construction.
3. `ProcessedStore` replaces the duplicated expression at:
   `etl/module.py:189`, `eda/module.py:239`, `features/module.py:28,179`,
   `training/module.py:52`, `evaluate/module.py:82`,
   `selection/imputation.py:174`, `reports/audit.py:667`,
   `tracking/report.py:22`, plus `loader.canonical_path:147`. Verified for taxi:
   ```
   etl ok run_dir data/processed canonical data/processed/taxi_canonical.parquet
   eda ok run_dir data/processed canonical data/processed/taxi_canonical.parquet
   amazon run_dir data/processed/amazon canonical data/processed/amazon/amazon_canonical.parquet
   ```
   Tests that build their own expected path from `cfg.environment` keep passing
   because the override is absent in their fixtures.
4. `discover.profile` now resolves the contract through the overlay (`config_path`).
   Consequence: a project-overlaid dataset name becomes re-profiled; a base-only
   checkout is unchanged. Two tests call `profile()` with a synthetic tree
   (`tests/test_profile.py:81`, `tests/test_qq.py:638`); their isolation now points
   `BROADWAY_CONFIG_OVERLAY_DIR` at the isolated tree instead of patching the
   loader's `CONFIGS_DIR` (patching that constant also redirects `viz.yaml` and
   breaks plotting). Test intent is unchanged; both pass.
5. `eda._provenance.sample_size` changes only when the configured budget exceeds the
   source rows (small datasets). Taxi is 8.5M rows > 50k budget → unchanged.
6. `onboard._render_plan_source` output text changes only for plans with no derived
   features. Compile-checked.
7. Ratchet: `ENFORCED` gains one series and `DECISION_ENFORCED` one pair; the pair
   is now enforced against a **validating record**, not an AST call. The test pins
   the enforced count at exactly 13.
8. Committed EDA evidence replaced (taxi → Amazon) in the shared pipeline slot — the
   only intentional user-visible replacement in this stage. The stale taxi
   `eda_correlations.png` in that slot is **deleted** by the writer's new
   `remove_figure` (the Amazon run draws no correlation panel); the dashboard now
   links only `eda_effects.png` and `eda_outliers.png` (verified).
9. Artifact scoping replaces the manual `git restore` of `reports/figures`: Amazon's
   discover evidence and Q-Q figures go to `artifacts/discover/amazon/`, its data
   manifest to `artifacts/tracking/amazon/`, and the evaluation dir to
   `artifacts/evaluation/amazon/`. Tracked `reports/figures/numeric_*.png` are never
   written by the Amazon run (verified: `git status reports/figures` empty after the
   profile run), so the committed taxi audit page stays self-consistent without a
   human step.
10. `discover` gains `--artifacts-subdir` so a re-bootstrap records the namespace
    instead of writing the shared discover slot/figures.
11. Local-only, not committed: `data/processed/amazon/*`,
    `artifacts/{discover,tracking,evaluation}/amazon/*`, `data/raw/amazon/*`
    (`data/` ignored).

Environment note: no MLflow server is running locally, so each step's decision-log
projection fails with a full traceback and logs
`mlflow projection failed for '<step>' ... — the step continues`. The JSON record is
the truth (`reports/decision_log.py` is a projection); the steps still exit 0. This
is pre-existing behaviour, not introduced here, and it is visible in the logs.

## 10. Tests and gates (verified)

Three focused public-behavior tests, one per new behavior:

- `tests/test_loader.py::test_contract_column_renames_are_applied_on_read` — a raw
  CSV with a trailing-space header is read through the loader; the frame carries the
  contract's labels, pandera's raw schema (the real consumer boundary) validates it,
  a `read_sample(columns=[contract name])` projection is translated back to the
  file's label, and an unbound rename target is rejected at parse.
- `tests/test_loader.py::test_dataset_processed_subdir_scopes_canonical_artifacts` —
  the dataset override outranks the environment default for `canonical_path`.
- `tests/test_loader.py::test_artifact_store_scopes_a_dataset_namespace` — the
  artifact override outranks the shared dir and an absolute/`..` subdir is rejected.
- `tests/test_experiment_outputs.py::test_decision_enforced_pair_needs_a_valid_record`
  — a decision-enforced pair is a violation until its `*_decision.json` validates
  (deleting the record fails the gate), and the enforced count is pinned at 13.

```
$ pytest tests/test_loader.py -q                    -> 14 passed
$ python scripts/check_experiment_outputs.py        -> EXPERIMENT OUTPUTS OK — 13 enforced steps, 23 advisory gaps (exit 0)
$ python scripts/check_project_paths.py --context dev -> PROJECT PATHS OK (exit 0)
$ ruff check src project tests scripts             -> All checks passed!
$ mypy src/broadway                                -> Success: no issues found in 199 source files
```

The `meta.head` parent-stamp probe is **fixed at the root**, not re-stamped by hand:
`agents/tools/render_gates.py` now derives `meta.head` from `git rev-parse HEAD` on
every render (and refuses to write a stamp when a rebase/merge/cherry-pick/revert is
in progress, or when git cannot answer), so the law holds by construction:
- while the render is uncommitted (dirty file) the probe's required value is HEAD;
- after the render's commit lands, the probe's required value is that commit's first
  parent — the same HEAD.
The round-trip test now renders a scratch registry and proves byte-identical output,
`meta.head == git HEAD`, and the digest header carrying that stamp. Because the
governance probe is **not** in the fast tier, the final gate is the full
`scripts/run_local_ci.sh` tier (§14). One pre-existing test-suite custody quirk:
`tests/test_ci_control_plane.py` runs `scripts/run_local_ci.sh --static`, whose
consolidation gate rewrites `artifacts/consolidation/duplicates.json`; `conftest`
hashes `artifacts/` at session start. Regenerating the ignored report before the
suite (as done) keeps the run green; the one duplicate group it reports is
pre-existing (taxi univariate `draw_full_forest`).

## 11. Risks and limits (what a reviewer should attack)

1. **Ordinal-as-regression** — allowed by the directive, flagged in §1. A model
   claim on this target would be unsound without an ordinal model; no model is
   trained or claimed.
2. **Selection bias in the survey step** — the metric's feature was chosen from
   EDA's ranking. Mitigated (not eliminated) by judging the record against the
   selection-adjusted null over the whole declared 12-feature family (0.1166, one
   shared permutation per draw) and asserting the record, sensitivity and report all
   carry that number; the interval's lower bound (0.4275) clears it by a wide margin.
   The derived value is stable across tie-handling variants (0.1166 midrank, 0.1176
   raw-code-vs-ranks, 0.1181 per-feature independent permutations) — all far below
   `min_effect` 0.15 — so the verdict does not depend on the variant. A
   pre-registered feature would be cleaner.
3. **EDA slot replacement** — committed taxi EDA evidence is overwritten; the git
   history retains it, but the working tree no longer shows the taxi run. Accepted
   under the retirement directive.
4. **Profile/audit split state** — the Amazon profile now lives in its own namespace
   (`artifacts/discover/amazon/`), so the tracked taxi `reports/audit/profile.md`
   plus `reports/figures/numeric_*.png` are untouched by construction. The unscoped
   `artifacts/discover/{profile,qq_overview}.json` and
   `artifacts/tracking/data_manifest.json` written by the pre-scoping Amazon run are
   deleted (ignored, regenerable) so a taxi `audit`/`tracking` render fails loud on
   a missing artifact instead of silently showing Amazon numbers. Stage 2 removes
   the taxi audit surface.
5. **Namespace scope** — `ProcessedStore` (processed data, 10 call sites) and
   `ArtifactStore` (discover/tracking/evaluation) are the only sanctioned owners; a
   future step that builds either path by hand would bypass them. Remaining
   **dataset-blind** surfaces are named, not hidden: the shared pipeline-evidence
   slot `project/experiments/results/pipeline/<step>/` (deliberately replaced this
   stage), the tracked `reports/{audit,lineage,tracking,results}/` render surfaces
   (owned by taxi's run), MLflow experiment/run names, `data/samples/` (named-sample
   registry), and `artifacts/{lineage,timeline}/` (dataset/analysis-scoped ids inside
   the files, single root on disk). Each disappears or is re-pointed in stage 2.
6. **Multi-select `Purchase_Categories`** — left as a 29-level categorical; EDA
   skipped it (sparse χ²). Splitting on `;` into indicators is a modelling decision
   for a later stage, not silently done here.
7. **n=602** — the EDA `min_sample_size` floor (100) is met, but per-level claims
   (rating 5: n=17; `Others` gender: n=19) are weak and shown with their n.
8. **`Timestamp` ignored** — 601 unique locale-format strings; parsing it as a
   survey window is future work, and it is deliberately not the datetime column.
9. **The MLflow projection is down locally** — records exist as JSON; the dashboard
   reads JSON. No MLflow state was written.
10. **Tie-uncorrected ε²** — the EDA value (0.2299) uses midranks without the tie
    correction in the denominator; the corrected variant on this family is 0.2515.
    The definition is now pinned in `src/broadway/eda/tests.py` and the record's
    `unit` carries `epsilon_squared`.

## 12. Stage 2 plan (write-only; nothing executed)

### 12.1 Untracked taxi assets → move to `/home/opc/ONE/ER-archive/` (never `rm`)

Pre-removal tag on the landed integration head (the tip this stage lands on, before
any deletion), created by the stage-2 owner:

```
git tag -a amazon-cutover-pre-taxi-removal -m "taxi assets archived before Amazon cutover"
git rev-parse amazon-cutover-pre-taxi-removal^{commit}   # record in the manifest
```

Archive manifest + checksum (one file beside the move, written before the move and
verified after): every moved path, its byte size and `sha256`, the tag commit, and
`du -sb` totals (~670 MB). The move is a `mv` into `ER-archive/taxi/<subtree>/`;
the manifest is re-checked with `sha256sum -c` after the move and the originals are
only removed once the manifest verifies.

Raw (`data/raw/`):
- `yellow_tripdata_2024-01.parquet` (48M), `-02.parquet` (49M), `-03.parquet` (58M)
- `taxi_zone_lookup.csv` (12K)

Processed (`data/processed/`):
- `taxi_canonical.parquet` (143M), `taxi_clean.json`, `taxi_join_audit.json`,
  `taxi_lookup_value_audit.json`
- `train.parquet` (107M), `val.parquet` (27M), `training_data.parquet` (103M),
  `train_features.parquet` (140M), `val_features.parquet` (36M),
  `feature_pipeline.pkl` (712K)
- `joined_sample_dev.parquet` (52K), `joined_sample_live.parquet` (3.0M),
  `sample_meta_dev.json`, `quality_report.json`
- Keep: `test_canonical.parquet`, `test_clean.json` (base synthetic test dataset),
  `deadcode_census.md` (census report, not taxi data), `amazon/` (this stage).

Samples (`data/samples/`):
- `fare_prediction_1m@v3.parquet` (22M) + `@v3.json`

Local-only state to archive with the data (or discard as regenerable, owner's call):
`mlruns/`, `.mlflow.db`, `project/mlruns/`, `checkpoints/` if taxi-bound.

Ignored machine state to clear (regenerable, never archived as data): the unscoped
`artifacts/` tree in full (discover/tracking/evaluation/lineage/consolidation/
timeline/experiments), and `data/processed/etl.*` + any `data/processed/*.json` a
taxi ETL run wrote at the processed root (the `etl.md/.png/_decision.json` evidence
files). Enumerate with `git status --ignored` before clearing.

### 12.2 Tracked files/dirs to remove or rewire

Config:
- `project/config/dataset/taxi.yaml`, `project/config/project/taxi.yaml`,
  `project/config/experiment/taxi.yaml`,
  `project/config/analysis/{taxi.yaml,taxi_hypothesis.yaml,taxi_causal.yaml}`
- base tree taxi knobs: `configs/step/features.yaml:14-16` (taxi column lists),
  `configs/step/train.yaml:9` (taxi feature surface), `configs/tooling.yaml`
  (taxi-referencing tool paths) — rewire to the Amazon binding or a generic default
- `project/config/sample/{fare_prediction_1m.yaml,taxi_diagnostic.yaml,taxi_estimation.yaml}`
- `project/config/slice/{airport,distance_duration_inconsistent,passenger_out_of_range,pre_2024}.yaml`
- `project/config/experiments/{fare_prediction,timeseries,multivariate,mlflow,working}.yaml`
  (rewire `mlflow.yaml` to the Amazon HPO spec or delete; `working.yaml` is taxi scratch)
- `project/config/feature_engineering.yaml`, `project/config/layout.yaml` (rewire paths)

Experiments + tracked results:
- `project/experiments/{fare_prediction,timeseries,more_modeling,multivariate,polynomial_regression_et_all,univariate,mlflow}/`
- `project/experiments/results/{fare_prediction,timeseries,more_modeling,multivariate,univariate,mlflow}/`
- `project/experiments/results/pipeline/eda/` — keep the Amazon evidence (already the
  Amazon run); delete nothing here.

Code:
- `project/data.py`, `project/working.py`, `project/ml_pipeline.py`,
  `project/features/__init__.py` (taxi plan) — keep `project/features/amazon.py`
- `project/scripts/01…12` (taxi narrative), `project/basic.py`, `project/boroughs.py`,
  `project/dashboard.py`, `project/experiments.py`, `project/config.py`,
  `project/STATS.md`, `project/k8s/optuna/configmap.yaml`
- `gpu_training/adapters/taxi.py` + its references in `gpu_training/config/*.yaml`

Tests:
- `project/tests/{test_dashboard.py,test_fare_temporal_recipe.py,test_ml_pipeline.py}`
- Rewire the taxi-coupled platform tests:
  `tests/test_column_plan.py` (taxi reference contract → Amazon contract),
  `tests/test_config.py:225` (taxi HPO spec), `tests/test_experiments_dispatcher.py:60`
  (`dataset=taxi` smoke), `tests/test_gpu_training.py` (taxi adapter),
  `tests/test_project_paths.py:59` (retired-config probe — keep, it pins the checker),
  `tests/test_recipe.py` ("taxi join shape" docstring — reword to the synthetic shape),
  `tests/test_branch_parity_scripts.py` (parity fixtures name taxi surfaces),
  `tests/test_tooling_consultant.py` (taxi tooling paths),
  `tests/test_mermaid.py` (taxi-named diagrams), `tests/test_profiling.py`
  (dataset="taxi" probe labels).
- `scripts/check_project_paths.py` retired-path regex list; `scripts/run_local_ci.sh`
  project-test step; `scripts/main_day_sync.sh` dev-only material list;
  `scripts/main_whitelist.txt` (taxi whitelist entries);
  and the taxi-referencing scripts `check_branch_parity.sh`, `check_docker_paths.sh`,
  `coldprobe_runner.py`, `promote_to_main.sh`, `pyright_advisory.sh`, `ship.sh`,
  `verify_live.py`.

Ignore rules / tracked surfaces the reviewer flagged:
- `.gitignore` taxi sample-evidence negations
  (`!project/experiments/results/univariate/sample_evidence/*`) and the
  `project/experiments/results/**/*.png` negation (kept for Amazon, re-scoped if the
  results tree moves); `reports/results/index.md` (walkthrough index naming taxi
  steps); ledger `index/*.yaml`, `DIGEST.md` and `factsheets/` entries naming taxi
  owners; `readmore/00_ingestion.md` (taxi ingest narrative).

Namespace question: `project/features/__init__.py` currently *is* the taxi plan
(module-level `CONTRACT = load_contract()` reading `dataset/taxi.yaml`). Removing it
turns `project/features` into a namespace package; decide explicitly between
(a) delete the module and keep `project/features/amazon.py` as the only plan, or
(b) keep the package with an `__init__.py` that exports the Amazon plan. Do not leave
a package whose import implies taxi.

Docs/maps: `README.md` (taxi examples, mode system D14, stats-script table),
`project.md`, `agents/ledger/HANDOFF.md` (taxi as the test case),
`agents/ledger/STATE.md`, `agents/ledger/INVENTORY.md`, `agents/ledger/DIGEST.md`
+ `agents/ledger/index/*` (re-render after the rows go), `reports/audit/*`,
`reports/lineage/*`, `reports/tracking/*`, `reports/results/index.md`,
`graphify-out/` (regenerate), `readmore/00_ingestion.md`.

### 12.3 Order and verification (stage 2)

1. Tag the integration head; move §12.1 assets into `ER-archive/` (verify by
   `diff -r` size/count before deleting the originals — reversible move only).
2. Remove/rewire §12.2 tracked files in one atomic change per group (config → code →
   tests → docs), keeping `data/processed/amazon/` and the Amazon configs untouched.
3. Re-run the ladder: `python -m project.cli etl/eda --dataset amazon`, the survey
   step, `scripts/check_experiment_outputs.py`, `check_project_paths.py --context dev`,
   ruff/mypy, `agents/tools/render_gates.py` (the stamp is generated, so the
   parent-stamp probe stays green), the FULL `scripts/run_local_ci.sh` tier, and the
   dashboard render.
4. Refresh `graphify` + `reports/lineage` so no map names a removed file.
5. Re-download gate: after the move, `sha256sum -c archive-manifest` and
   `git status --short` must both be clean before the stage-2 commit lands.

### 12.4 Rollback

- Stage 1: `git revert <commit-A> <commit-B> <commit-C>` restores the platform read
  boundary and removes the Amazon binding; committed taxi EDA evidence returns from
  history; `data/processed/amazon/` and `data/raw/amazon/` are ignored/regenerable and
  can be deleted without loss.
- Stage 2: the pre-removal tag plus `ER-archive/` are the restore path; no `rm` on
  tracked or archived data.

## 13. Open questions for the owner

1. Ordinal target: keep `regression` (this stage, per the directive) or declare
   `classification` for the modelling stages? The association question is
   unaffected; the model choice is not.
2. `Purchase_Categories` — split into indicator columns in a later feature stage, or
   keep as a frequency-encoded categorical?
3. `Timestamp` — leave ignored, or parse it in stage 2 as a survey-window block for
   block-resampled intervals (it has 601 unique values)?
4. Should `mlflow` HPO config be rewired to Amazon or dropped in stage 2?
5. Stage-2 asset archive path/owner confirmation for the ~670 MB of taxi parquet
   (move, never delete).

## 14. Gate evidence (appendix)

```
$ python -m project.cli etl --dataset amazon --experiment amazon   # exit 0
$ python -m project.cli eda --dataset amazon                       # exit 0
$ python project/experiments/amazon_survey/01_profile.py           # exit 0
$ python scripts/check_experiment_outputs.py                       # exit 0 (13 enforced steps)
$ python scripts/check_project_paths.py --context dev              # exit 0
$ ruff check src project tests scripts                             # All checks passed!
$ mypy src/broadway                                                # Success: 199 files
$ pytest tests/test_governance_probes.py -k head_parent_stamp      # green at the final tip
$ bash scripts/run_local_ci.sh                                     # FULL tier (tail in the report)
$ agents/tools/render_gates.py                                     # stamp generated from HEAD
```

## 15. Revision log (adversarial review, round 1 — REVISE)

| # | Reviewer finding | Fix | Evidence |
|---|---|---|---|
| 1 | stale taxi `eda_correlations.png` linked beside the Amazon record | `EdaArtifactWriter.remove_figure` unlinks a same-stem figure this run did not draw; the correlation status is recorded in the summary provenance | PNG deleted (`git status` D); the dashboard links only effects/outliers |
| 2 | `meta.head` hand-maintained, probe red on the pushed tip | the stamp is **generated** by `agents/tools/render_gates.py` from `git rev-parse HEAD` on every render; refuses a rewrite in progress; the round-trip test proves byte-identical output + `meta.head == HEAD` | probe `-k head_parent_stamp` green at the final tip; full tier |
| 3 | record judged against the per-feature null (0.0771) while claiming 0.095; family was 3 of 12 | one shared target permutation per draw over the config-declared 12-feature family; `model_copy(null_threshold=family_null)` before `to_decision`; runtime assert record == published sensitivity < `min_effect` | record/report/falsifier all 0.116589 |
| 4 | `read_sample(columns=[contract name])` raised `ColumnNotFoundError` | projection translated through `ColumnNormalizer.source_name` before `select` | pinned in the rename test |
| 5 | artifact outputs still dataset-blind; manual `git restore` was the mechanism | `ArtifactStore` + `artifacts_subdir` scope discover/tracking/evaluation; the audited reads use the same store; remaining blind surfaces are named | `git status reports/figures` empty after the Amazon profile; taxi paths unchanged |
| 6 | `DECISION_ENFORCED` only checked the AST call; deleting the record passed | the pair now needs a `Decision.model_validate_json` record; the enforced count is pinned at 13 | new focused test |
| 7 | the Amazon surface was absent from the ledger | `GATE-INFRA-183` with owner paths, inputs/outputs, the 12-feature/namespace rule and one finding; DIGEST + index re-rendered | registry tests + full tier |
| 8 | column literals and a hardcoded quantile/CI in the step | target from `cfg.dataset.target`, feature/family from the experiment config, segments from `SegmentPlan.build`, quantile from the rule, CI multiplier from the confidence | no dataset literal left in the script |
| 9 | the plan frequency-encoded the 1–5 rating columns | the ordinals are numeric `source` keeps; the hand-fix is recorded in the plan docstring | plan inspected (`source` int64 ops) |
| 10 | stage-2 plan missed several surfaces | §12 now lists ignore rules, `reports/results/index.md`, ledger index/DIGEST/factsheets, `readmore/00_ingestion.md`, the taxi-referencing scripts/whitelist, the four missed tests, base config taxis, the namespace-package question, and the tag + archive manifest/checksum | §12 |


## 16. Stage-2 completion note (taxi removal executed 2026-11-27)

Status: **executed, not pushed.** Owner approved the full replacement; this stage ran
the §12 plan in the mandated order (preserve → remove/rewire → rewire to Amazon → nit).

### 16.1 Preserve (reversible)

- Tag `pre-amazon-removal` at `d376f634f57d5b2aaa242ca07c65eba6527cfffc` (the stage-1
  integration head), message "taxi dataset + experiments before Amazon replacement".
- Untracked taxi data moved (never `rm`) to
  `/home/opc/ONE/ER-archive/broadway-taxi-data-2026-11-27/` — **25 files, ~790 MB**,
  `sha256sum -c MANIFEST.sha256` verifies. Includes `data/raw/yellow_tripdata_2024-0{1,2,3}.parquet`,
  `taxi_zone_lookup.csv`, the processed canonical/train/val/feature parquet + clean/audit
  JSON + feature pipeline, the `fare_prediction_1m@v3` sample, non-Amazon
  `artifacts/evaluation/*` and the stale taxi `artifacts/reports/eda.html`. A README
  records what/why/how to restore. No file was open (`lsof`/`fuser` clean).
- `data/raw/amazon/` and `data/processed/amazon/` were not moved; the raw CSV sha256 is
  unchanged (`edb1af97…`). ETL/EDA reruns regenerated `data/processed/amazon/` as expected.

### 16.2 Removed / rewired (recoverable via tag + history)

- Config: taxi dataset/analysis/experiment/project/sample/slice configs, the taxi
  experiment series configs (`fare_prediction`, `timeseries`, `multivariate`, `mlflow`,
  `working`), and the project `feature_engineering.yaml`.
- Experiments: `fare_prediction`, `timeseries`, `more_modeling`, `multivariate`,
  `polynomial_regression_et_all`, `univariate`, `mlflow` (scripts + tracked results).
  `amazon_survey` and the Amazon pipeline-EDA evidence are kept.
- Code: `project/{data,working,ml_pipeline,config,experiments,basic,boroughs}.py`,
  `project/scripts/*`, `project/STATS.md`, `project/k8s/optuna/configmap.yaml`.
  `project/features/__init__.py` is now an import-light package file (the Amazon plan is
  `project/features/amazon.py`).
- Tests: taxi-only project suites removed; platform tests rewired to platform-only or
  Amazon (`test_column_plan`, `test_config`, `test_experiments_dispatcher`,
  `test_gpu_training`, `test_project_paths`, `test_recipe`, `test_mermaid`,
  `test_profiling`). The retired-config probe and the branch-parity/tooling tests are
  retained (they pin branch vocabulary / the checker, not taxi data).
- GPU lane: `gpu_training/adapters/taxi.py` → `project.py` (objective
  `adapters.project:build`), study config points at `dataset: amazon`,
  `experiment: amazon`, `hpo_spec: project/config/experiments/amazon_hpo.yaml` (new).
- Base configs: taxi imputation columns moved out of `configs/step/features.yaml` into
  `project/config/step/features.yaml` (Amazon columns); taxi comments removed from
  `configs/step/train.yaml` / `configs/tooling.yaml`.
- `.gitignore`: univariate sample-evidence negations dropped; Amazon evidence negations kept.
- Ledger: retired rows (`GATE-TRAIN-32`, `GATE-CFG-104/111/112`, `GATE-INFRA-122/129/180/181`,
  `GATE-CUST-149/152`) recorded in `retired:`; live GPU/contract rows updated to the
  Amazon/`project.py` naming; `GATE-INFRA-183` updated for the single-dataset state.
  `DIGEST.md` + `index/*` re-rendered (round-trip byte-identical).
- Docs/maps: README/project.md/TODO/CONTRIBUTING/HPO_TRAINING/readmore rewritten to the
  Amazon project; stale taxi report renders and `diagrams/data_tracking.mmd` removed;
  `reports/lineage` citation dropped from the contract (`src/broadway/lineage/graph.py`
  remains the code surface).
- CI: the removed `project/experiments.py verify` smoke step was dropped; workload-image
  steps already auto-skip when `project/config/experiments/mlflow.yaml` is absent.

### 16.3 The known nit (fixed)

`project/experiments/amazon_survey/01_profile.py` no longer hardcodes the dataset name or
the 0.95 CI level: the dataset is `dataset:` in `project/config/experiments/amazon_survey.yaml`,
the contract name is `cfg.dataset.name`, and the CI multiplier comes from
`measurement.confidence`. The regenerated record is byte-identical (same
`0.5015 [0.4275, 0.5729]`, null `0.1166`, n=602, verdict `act`).

### 16.4 Verification (real artifacts)

- `python -m project.cli etl --dataset amazon --experiment amazon` → exit 0.
- `python -m project.cli eda --dataset amazon` → exit 0 (602 rows, same decisions).
- `python project/experiments/amazon_survey/01_profile.py` → exit 0, record unchanged.
- `scripts/check_experiment_outputs.py` → `EXPERIMENT OUTPUTS OK — 1 enforced steps,
  0 advisory gaps` (the taxi series are gone; the count is no longer 13 by construction).
- `scripts/check_project_paths.py --context dev` → OK.
- `pytest tests/test_governance_probes.py -k head_parent_stamp` → green.
- `bash scripts/run_local_ci.sh` (FULL) → `LOCAL-CI GREEN` (parity, ruff, mypy,
  pyright-advisory, vulture, configs, shell-scripts, docker-paths, deps, consolidation,
  logging, experiment-outputs, pytest, project-tests).
- `git grep -il taxi` now lists only historical arbitration/archive/factsheet/notes
  records, branch-vocabulary files (`taxi` is the branch name), the retired-path probe,
  and the ledger's preserved retired rows.

### 16.5 Remaining risks

1. **K8s HPO workload is parked, not re-homed.** The taxi-bound `k8s/optuna` worker and
   `project/k8s/optuna/configmap.yaml` are gone; the generic k8s/optuna manifests and
   Dockerfiles remain and auto-skip because `project/config/experiments/mlflow.yaml` is
   absent. Distributed HPO for Amazon is future work (documented in HPO_TRAINING.md).
2. **The `git grep` residue is intentional** (history + branch name). The dataset
   references inside retired ledger rows are preserved as history, not live ownership.
3. **`project/config/step/features.yaml` overlays every project step load** (stage-1
   design), so it also applies when a project checkout loads the synthetic `test` dataset;
   the imputation study validates its columns against the plan and fails loud, so this is
   visible rather than silent.
