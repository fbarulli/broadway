"""Shared paths, constants, and dataset loaders for this experiment (no analysis logic)."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parents[2] / "data" / "raw"
RESULTS = HERE.parents[2] / "experiments" / "results" / HERE.parents[0].name / HERE.name
CLEAN_PARQUET = RESULTS / "sample50k.parquet"
FULL_PARQUET = RESULTS / "full_sample.parquet"
RATECODE1_PARQUET = RESULTS / "ratecode1_sample.parquet"

# The single working dataset all analysis steps operate on from now on.
# sample50k / full_sample are retired (steps 01-10 kept as history).
WORKING_DATASET = RATECODE1_PARQUET

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
                "fare_amount >= 0.0",
                "trip_distance >= 0.0",
                "trip_duration > 0",
            ],
        },
    },
}


def load_metered() -> pd.DataFrame:
    """Ratecode1 trips with duration_minutes derived; non-positive durations dropped."""
    df = pd.read_parquet(RATECODE1_PARQUET)
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    df["duration_minutes"] = df["trip_duration"] / 60
    return df[df["trip_duration"] > 0]
