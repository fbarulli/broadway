# tests/

## test_config.py
Verifies YAML → Pydantic config loading for every step type. Ensures missing
or invalid configs raise the correct errors.

## test_cli.py
CLI dispatch tests: discover generates YAML from CSV, train dispatches to
pipeline, missing subcommand and invalid step raise argparse errors.

## test_contracts.py
Data contract checks against real data: columns, dtypes, and nulls are
validated per configs/dataset/test.yaml. Tests missing columns, wrong dtypes,
and nulls above the config threshold.

## test_eda.py
EDA module tested against synthetic data: summarize, quality checks (constant
columns, duplicates, outliers), and missingness analysis (null counts, patterns).
