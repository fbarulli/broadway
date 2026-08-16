"""Shared paths/constants for this experiment (no logic)."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parents[2] / "data" / "raw"
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name
CLEAN_PARQUET = RESULTS / "sample_50k.parquet"
FULL_PARQUET = RESULTS / "full_sample.parquet"
RATECODE1_PARQUET = RESULTS / "ratecode1_sample.parquet"

# Per-dataset provenance: how the dataset was built (transformations) and the
# X/Y it is analyzed with. Keyed by parquet stem; used by _tests.py to make
# every tests_<dataset>.json self-describing.
DATASET_META = {
    "sample_50k": {
        "x_columns": ["trip_distance"],
        "y_column": "fare_amount",
        "transformations": {
            "sample": {
                "method": "random",
                "size": 50000,
                "seed": 42,
                "source": "project.data.read_training_sample",
            },
            "filters": [
                "fare_amount > 2.50",
                "trip_distance > 0.0",
                "trip_distance <= 50.0",
            ],
        },
    },
    "full_sample": {
        "x_columns": ["trip_distance"],
        "y_column": "fare_amount",
        "transformations": {
            "sample": {
                "method": "random",
                "size": 50000,
                "seed": 42,
                "source": "project.data.read_training_sample",
            },
            "filters": [
                "fare_amount > 2.50",
                "trip_distance > 0.0",
                "trip_distance <= 50.0",
            ],
        },
    },
    "ratecode1_sample": {
        "x_columns": ["trip_distance"],
        "y_column": "fare_amount",
        "transformations": {
            "sample": {
                "method": "random",
                "size": 50000,
                "seed": 42,
                "source": "raw yellow_tripdata_2024-01..03 parquets",
            },
            "filters": [
                "RatecodeID == 1",
                "fare_amount >= 0.0",
                "trip_distance >= 0.0",
            ],
        },
    },
}
