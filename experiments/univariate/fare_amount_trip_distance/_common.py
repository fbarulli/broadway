"""Shared paths/constants for this experiment (no logic)."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name
CLEAN_PARQUET = RESULTS / "sample_clean.parquet"
FULL_PARQUET = RESULTS / "full_sample.parquet"
TESTS_JSON = RESULTS / "tests.json"
