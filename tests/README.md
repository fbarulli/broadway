# tests/

Test suite organized by capability. Tests use contract-driven valid fixtures
plus explicit malformed fixtures, so they are not tautological. The
non-taxi onboarding E2E test (`test_onboarding_e2e.py`) is the acceptance
proof that `src/broadway/**` needs no edits when the dataset is swapped.

## Config & contracts

YAML → Pydantic loading and dataframe contract enforcement.

- `test_config.py` — every step type loads from YAML; missing/invalid config raises.
- `test_contracts.py` — generated-data columns/nulls validated per `configs/dataset/test.yaml` (never real data).
- `test_pandera_schemas.py` — raw schema generated from `DatasetContract.columns`.
- `test_analysis_contract.py` — `AnalysisContract` validation + `require_mode` guardrails.
- `test_training_contracts.py`, `test_evaluate_contracts.py` — `TrainingResult` / `EvaluationResult` models.

## Ingestion / ETL / data

Ingest, structural cleaning, joins, and the dataset loaders.

- `test_process.py` — ETL function units (`filter_valid_trips`, `compute_trip_duration`, …) on synthetic frames.
- `test_structural_cleaning.py` — `standardize_missing`, `parse_datetime`, duplicate/target-null removal, `StructuralCleanResult`.
- `test_join_audit.py`, `test_lookup_value_audit.py` — `JoinAudit` / `LookupValueAudit` evidence.

> **Taxi-layer tests** (`project/tests/`) cover the taxi ETL (`project/etl/`)
> and project config consistency with generated data. Platform tests never
> touch project-level data or configs — enforced by `test_platform_hygiene.py`.

## Lineage

Run graph, sidecar records, and sample specs.

- `test_lineage_models.py` — `LineageRecord` / `LineageGraph` / `RunState` models.
- `test_lineage_graph.py`, `test_lineage_mermaid.py`, `test_lineage_state.py` — graph assembly, Mermaid output, run-state derivation.
- `test_sample.py` — `SampleSpec` loading + `column_mapping`.

## Timeline / walkthrough

`src/broadway/timeline/` plus the `reports/results.py` / `reports/timeline.py` renderers.

- `test_timeline.py` — `AnalysisStep`/`AnalysisDecision`/`Suggestion` models, persistence roundtrips, sequence config, and `render_timeline` status vocabulary.
- `test_walkthrough.py` — walkthrough orchestration: decision-gate stops, idempotent resume, `--force` rerun, runner evidence, and failure capture.
- `test_decide.py` — `decide.record` validation (method allowlist per kind, unknown-kind rejection).
- `test_suggest.py` — deterministic suggestion layer (`suggest_after`/`suggest_next`) + `render_dashboard`.
- `test_results.py` — `render_results`/`write_results`, humanization (3 sig figs, p-value floor), orphan deletion, failed-step page omission.
- `test_figures.py` — `FigureRef` captions render as figures in `timeline.md` (`figures/...`) and results pages (`../figures/...`); `run_describe`/`run_normality` attach figures.

## Stats library

`src/broadway/stats/` (pandas/numpy only).

- `test_base.py`, `test_plan.py` — stratified sampling + `AnalysisPlan` (de)serialization.
- `test_effect_size.py`, `test_assumptions.py`, `test_anova.py`, `test_post_hoc.py` — eta²/omega², Levene/normality, ANOVA/Welch/Kruskal, Games-Howell.
- `test_regression.py`, `test_diagnostics.py`, `test_time_series.py`, `test_baseline.py` — OLS/robust SE, residual diagnostics, DW/ACF, LightGBM baseline.
- `test_describe.py`, `test_stats_module.py` — describe result + the `stats run` step.

## Discover / profile / onboarding

- `test_profile.py` — `DatasetProfile` / `ColumnProfile` observed facts.
- `test_qq.py` — `plot_numeric_qq` small multiples + per-feature distribution grid (exclusion/chunking), `run_normality` joint per-group Q-Q cap, and profile-evidence rendering.
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
- `test_loud_failures.py` — silent-failure regressions (missing/malformed artifacts, non-finite metrics).
- `test_reports.py` — `reports/` renderers (`index.md`, per-step markdown).
- `test_surface_integrity.py` — tracked `reports/` markdown link resolution + 5 MB HTML / 2 MB PNG size caps (read-only).
