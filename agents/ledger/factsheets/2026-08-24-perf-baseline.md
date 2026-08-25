# FACT SHEET — PERF-BASELINE (D23 recorder + phase IO map) — 2026-08-24

Investigator: read-only static lane PERF-BASELINE @ HEAD 5016e93
(sklearn). No execution; static analysis only.

## Recorder verdict: ABSENT — confirmed
No script/module/CI step records D23 metrics M1–M4 anywhere. Evidence:
greps over *.py/*.sh/*.yml for M1=/M2=/cold_bb_s/COLDPROBE/cache_hit,
telemetry, static.gate/cold.bb/duplication.overhead → hits ONLY in
prose (DECISIONS.md:268-286 D23 text itself; FIXES.md:147 incident
prose). STATE.md has NO ## telemetry section (headers enumerated).
Only timing code in src/: trainer.py:62-68 train_time_seconds
(per-model-fit scope, not M1-M4). ci.yml jobs platform/build-and-boot/cd
capture no timings; scripts/ inventory = parity/champion/e2e-det/
run_local_ci/ship/tier_classifier only. artifacts/baseline/baseline.json
is an ML prediction baseline (mean MAE 118.12), NOT performance telemetry.
D23 senior clauses (DECISIONS.md:283-286) name "recorder = senior agent,
rows into STATE.md telemetry"; COLDPROBE-1 "mandatory" — never built.

## Phase map → IO table (configs/flow/prediction.yaml:
discover→etl→contracts→baseline→features→train→evaluate)

| # | Phase | READ | WRITE |
|---|---|---|---|
| 0 | ingest process.py:194-218 (CLI cli.py:143-146) | all raw parquets full polars→pandas collect :36-40; CI 50k cap :43-48 | processed/training_data.parquet :132-137; lineage JSONs :211-217 |
| 1 | etl module.py:61-177 | training_data.parquet FULL (read A) loader.py:124-129; lookup CSV ×2 :134; CI 50k sample :74-78 | taxi_canonical.parquet :98; train/val parquet :104-105 (time split val .2); taxi_clean.json + audit JSONs :135-166 |
| 2 | contracts module.py:24 | training_data.parquet re-read B (full O(N) for column/null checks) | none |
| 3 | baseline module.py:54,61 | same file re-read C (mean/median) | artifacts/baseline/baseline.json |
| 4 | features module.py:22-55 | train+val parquet :25-27 | train/val_features.parquet :45,:50 (~150 MB); feature_pipeline.pkl :52-54 |
| 5 | train module.py:89-158 | features parquets :42-44 (read #2 of features); HPO 50 trials hpo.py:159 | training_result.json :151 (train_time_seconds); MLflow run |
| 5b | (unwired) log_dataset mlflow_utils.py:92-129 | would be full-frame read D when wired; today tests only; backlog FIXES.md:53-54 | — |
| 6 | evaluate module.py:82-196 | val_features :49 + train_features :58 (#3); candidate+champion pyfunc :75,:79; 5-fold CV on full train :140-147 (cv_folds:5) | artifacts/evaluation/metrics.json :165 (+rewrite :191) |

Side flows: pipeline skips discover (pipeline.py:16-18); stats reads
canonical OR sample; hypothesis walkthrough reads canonical full without
--sample (timeline/runners.py:98-104); samples/generate reads full then
draws fixed size.

Re-read tally (local full-data prediction run): same logical data crosses
disk boundary ≥8× — training_data ×3 (A,B,C), train/val ×1, features ×3 —
plus ~5 large writes. Row-count "8.5M" attested by prose only (raw =
3 parquets ~160MB compressed; count UNVERIFIED statically).

## Dominance ranking (STATIC expectation, local mode)

1. train/HPO — O(trials × N_train) compute-bound (50 optuna trials lgbm)
2. evaluate CV — ≈4-5 extra full fits + third features read
3. etl — O(N) pandas chain with ≥4 full-frame copies (process.py:58,75,89)
   + ~290 MB parquet writes
4. contracts + baseline — two redundant full O(N) reads, near-zero compute
   (pure IO waste; cheapest fix)
5. features — vectorized O(N) + ~150 MB writes
6. ingest — one-time polars collect→pandas
7. stats/walkthrough/samples — sampled/fixed-size; minor

CI note: under CI=true everything downstream of etl sampling is O(50k).

## Requirements for a future timed-baseline lane (spec only)

M1 wrap run_local_ci.sh fast tier behind timer · M2 isolate pytest+cov
invocation behind timer · M3 Actions API job-timestamp derivation
(no gh api usage exists in repo today — machinery must be built) ·
M4 build-and-boot wall time filtered to cache misses, cold floor n≥3 ·
window = rolling 20-green sklearn runs, cancelled logged separately ·
COLDPROBE-1 gated on XDIST-1b + ≥5 greens · landing surface = new
STATE.md ## telemetry section (must be created) + DECISIONS D23 addendum.
