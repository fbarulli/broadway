# tests/

Test suite organized by capability. Tests use contract-driven valid fixtures
plus explicit malformed fixtures, so they are not tautological. The
non-taxi onboarding E2E test (`test_onboarding_e2e.py`) is the acceptance
proof that `src/broadway/**` needs no edits when the dataset is swapped.

## Config & contracts

YAML → Pydantic loading and dataframe contract enforcement.

- `test_config.py` — every step type loads from YAML; missing/invalid config raises.
- `test_contracts.py` — real-data columns/dtypes/nulls validated per `configs/dataset/taxi.yaml`.
- `test_pandera_schemas.py` — raw schema generated from `DatasetContract.columns`.
- `test_analysis_contract.py` — `AnalysisContract` validation + `require_mode` guardrails.
- `test_training_contracts.py`, `test_evaluate_contracts.py` — `TrainingResult` / `EvaluationResult` models.

## Ingestion / ETL / data

Ingest, structural cleaning, joins, and the dataset loaders.

- `test_process.py` — ETL function units (`filter_valid_trips`, `compute_trip_duration`, …) on synthetic frames.
- `test_taxi_data.py` — mode-keyed cache + stratified/time-slice loaders against real data.
- `test_structural_cleaning.py` — `standardize_missing`, `parse_datetime`, duplicate/target-null removal, `StructuralCleanResult`.
- `test_join_audit.py`, `test_lookup_value_audit.py` — `JoinAudit` / `LookupValueAudit` evidence.

## Lineage

Run graph, sidecar records, and sample specs.

- `test_lineage_models.py` — `LineageRecord` / `LineageGraph` / `RunState` models.
- `test_lineage_graph.py`, `test_lineage_mermaid.py`, `test_lineage_state.py` — graph assembly, Mermaid output, run-state derivation.
- `test_sample.py` — `SampleSpec` loading + `column_mapping`.

## Stats library

`src/broadway/stats/` (pandas/numpy only).

- `test_base.py`, `test_plan.py` — stratified sampling + `AnalysisPlan` (de)serialization.
- `test_effect_size.py`, `test_assumptions.py`, `test_anova.py`, `test_post_hoc.py` — eta²/omega², Levene/normality, ANOVA/Welch/Kruskal, Games-Howell.
- `test_regression.py`, `test_diagnostics.py`, `test_time_series.py`, `test_baseline.py` — OLS/robust SE, residual diagnostics, DW/ACF, LightGBM baseline.
- `test_describe.py`, `test_stats_module.py` — describe result + the `stats run` step.

## Discover / profile / onboarding

- `test_profile.py` — `DatasetProfile` / `ColumnProfile` observed facts.
- `test_onboard.py`, `test_init.py` — semantic inference hints + `ds-pipeline init` scaffolding.
- `test_onboarding_e2e.py` — non-taxi CSV: init → etl → contracts → baseline → features → train → evaluate with a local MLflow file store.

## Modeling / features / baseline

- `test_ml_pipeline.py` — `FeaturePipeline` fit/transform/fit_transform.
- `test_builders.py`, `test_generic_features.py` — feature-builder registry + generic datetime/categorical builders.
- `test_baseline_module.py` — mode-dispatched `BaselineResult`.

## Causal

- `test_causal.py`, `test_causal_design.py`, `test_causal_module.py` — experiment design, power/MDE, assignment, and the `causal` step.

## Pipeline lifecycle / dispatch

- `test_full_dispatch.py` — `full` resolves the mode-specific flow.
- `test_lifecycle.py` — end-to-end chain wiring.

## CLI / integration / soundness

- `test_cli.py` — dispatch + argparse errors.
- `test_integration.py` — synthetic end-to-end: load → clean → split → train → evaluate.
- `test_imports.py` — all modules import; no stale references.
- `test_eda.py` — summarize / quality / missingness.
- `test_loud_failures.py` — silent-failure regressions (missing/malformed artifacts, non-finite metrics).
- `test_reports.py` — `reports/` renderers (`index.md`, per-step markdown).
