# tests/

## test_imports.py
Verifies all broadway modules import without errors and no stale `logistics_ml`
references remain. Uses `load_config()` to pass validated config values to
constructors (e.g. `FeaturePipeline`).

## test_config.py
Verifies YAML → Pydantic config loading for every step type. Ensures missing
or invalid configs raise the correct errors.

## test_cli.py
CLI dispatch tests: discover generates YAML from CSV, train dispatches to
pipeline, missing subcommand and invalid step raise argparse errors.

## test_contracts.py
Data contract checks against real data: columns, dtypes, and nulls are
validated per configs/dataset/taxi.yaml. Tests missing columns, wrong dtypes,
and nulls above the config threshold.

## test_eda.py
EDA module tested against real data: summarize, quality checks (constant
columns, duplicates, outliers), and missingness analysis (null counts, patterns).

## test_process.py
Unit tests for individual ETL functions (`filter_valid_trips`,
`compute_trip_duration`, `rename_columns`, etc.) using synthetic DataFrames.
All thresholds read from `process_config` (Pydantic-validated YAML).

## test_ml_pipeline.py
Unit tests for `FeaturePipeline`: fit learns route encodings and stats,
transform produces all engineered features, fit_transform combines both,
and transform-before-fit raises `RuntimeError`.

## test_integration.py
End-to-end test on synthetic data: load → clean → split → train → evaluate.
Verifies the full pipeline chain works without real data or infra dependencies.
