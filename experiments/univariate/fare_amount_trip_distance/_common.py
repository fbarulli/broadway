"""Shared paths, constants, and dataset loaders for this experiment (no analysis logic).

The working-dataset binding (parquet path, filters, loaders, time_bucket) is
OWNED by `project.working` (single source of truth, config-driven) and
re-exported here so the step scripts keep their `from _common import ...`
imports unchanged. Only experiment-layout paths and per-dataset evidence
provenance (DATASET_META) stay local.
"""

from pathlib import Path

from project.working import (
    MAX_DURATION_MINUTES,
    MIN_TARGET_VALUE,
    WORKING_DATASET,
    load_metered,
    load_working,
    time_bucket,
)

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parents[2] / "data" / "raw"
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name
CLEAN_PARQUET = RESULTS / "sample50k.parquet"
FULL_PARQUET = RESULTS / "full_sample.parquet"
RATECODE1_PARQUET = WORKING_DATASET  # kept as the historical alias

# Per-dataset provenance: how the dataset was built (transformations) and the
# X/Y it is analyzed with. Keyed by parquet stem; used by _tests.py to make
# every <dataset>.json self-describing.
DATASET_META = {
    "sample50k": {
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
        "x_columns": ["trip_distance", "duration_minutes"],
        "y_column": "fare_amount",
        "transformations": {
            "sample": {
                "method": "random",
                "size": 50000,
                "seed": 42,
                "source": "raw yellow_tripdata_2024-01..03 parquets",
            },
            "derived": {
                "duration_minutes": "(tpep_dropoff_datetime - tpep_pickup_datetime).total_seconds() / 60"
            },
            "filters": [
                "RatecodeID == 1",
                "fare_amount > 2.50 (voided rides)",
                "trip_distance >= 0.0",
                "trip_duration > 0",
                "duration_minutes < 240",
            ],
        },
    },
}
