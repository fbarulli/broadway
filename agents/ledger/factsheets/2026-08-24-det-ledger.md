# FACT SHEET — DET-LEDGER (determinism & reproducibility) — 2026-08-24

Investigator: read-only lane DET-LEDGER @ HEAD 5016e93 (sklearn).
Citations are working-tree file:line.

## Item verdicts

| ITEM | STATUS | EVIDENCE |
|---|---|---|
| ≥3 independent RNG families | CONFIRMED | A) YAML `random_state=42`: configs/experiment/{baseline:17,taxi:47,engineered:24,hyperopt:21}.yaml, configs/step/{train:1,etl:3,viz:8}.yaml; carriers src/broadway/config/schema.py:192,228,278 + config/viz.py:68; feeds etl/module.py:73-103, data/splitter.py:11-40, evaluate/validation.py:32-38, training/trainer.py:36-37, hpo.py:140,223, optuna.py:69,89, evaluate/explain.py:29-56, discover/qq.py:601-613, causal/assignment.py:11,36. B) sample-spec seed=42: configs/sample/fare_prediction_1m.yaml:9 → samples/generate.py:119 (seed recorded :134); experiments/mlflow/_common.py:69,95-120; multivariate/_setup.py:109-112; project/data.py:106. C) hardcoded literals: stats/assumptions.py:29 rng(0); onboard/module.py:163 =42; discover/qq.py:820 global np.random.seed(42); experiments/more_modeling/* literals |
| uv.lock dual numpy | CONFIRMED | numpy 2.3.5 uv.lock:2010-2031 (darwin+x86_64 markers) AND numpy 2.5.2 :2033-2044 (negation markers incl. win32); dep rows :584-585 |
| Shapiro subsample seeded | YES but literal | stats/assumptions.py:26-30: >5000 → default_rng(0).choice(5000, replace=False); test asserts cap size only, not indices (tests/test_assumptions.py:36-82) |
| Golden-float ULP unfixed | CONFIRMED | project/tests/test_ml_pipeline.py:153-163 bare == on dtypes/lists/records; backlog note FIXES.md:49-52 |
| wall-clock / order hazards | PARTIAL | created_at persisted: samples/generate.py:146→148-150, baseline/module.py:30-36,81-85, timeline/runners.py:58 + decide.py:44; trainer.py:62-64 time.time() log-only (persist UNVERIFIED). No unsorted FS walks in src/; sole glob consumer sorts (process.py:30). groupby sort=True (features/transformers.py:57; stats/post_hoc.py:14). Unseeded draw: experiments/more_modeling/15_complexity_funnel.py:65 (PNG only) |

## Byte-for-byte BY CONSTRUCTION

sample parquets (generate.py:119,126 + self-recorded sha256 :145) · splits
(splitter.py:12,25,39-40; seeded KFold validation.py:38; TimeSeriesSplit :34) ·
model fits via cfg.experiment.random_state (trainer.py:36-37; registry
:36,50,64) · HPO constructor-seeded TPESampler + per-worker offsets
(hpo.py:132-140,207-223; optuna.py:59-69) · ETL CI subsample ·
Shapiro/skew/kurtosis >5000 groups · SHAP bg + permutation importance · QQ
subsample · multivariate sample+shuffle · project ETL glob ordering.

## Silence list (law says nothing)

(a) assumptions.py:29 seed literal 0 — absent from every YAML/schema/artifact;
no index pinning test. (b) onboard/module.py:163 literal 42 bypasses the
"YAML single source of truth" rule (trainer.py:29). (c) qq.py:820 mutates
GLOBAL legacy numpy RNG state. (d) Wall-clock bytes embedded in artifacts;
no freeze flag exists. (e) Dual-numpy lock ⇒ cross-platform float divergence
while golden test keeps exact ==. (f) Experiment plot scripts unseeded /
hardcoded literals instead of config. (g) trainer timing persist-path
UNVERIFIED. (h) structural.py unique() warning examples assume input row order.
