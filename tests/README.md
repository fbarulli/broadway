# tests/

## test_imports.py
Verifies all broadway modules import without errors and no stale `logistics_ml`
references remain. Uses `load_config()` to pass validated config values to
constructors (e.g. `FeaturePipeline`).

## test_config.py
Verifies YAML → Pydantic config loading for every step type. Ensures missing
or invalid configs raise the correct errors.

## test_process.py
Unit tests for individual ETL functions (`filter_valid_trips`,
`compute_trip_duration`, `rename_columns`, etc.) using synthetic DataFrames.
All thresholds read from `process_config` (Pydantic-validated YAML).

## test_integration.py
End-to-end test on synthetic data: load → clean → split → train → evaluate.
Verifies the full pipeline chain works without real data or infra dependencies.
