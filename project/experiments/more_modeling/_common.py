"""Shared paths, constants, and loader for more_modeling experiments.

The sample is declared once in ``project/config/sample/fare_prediction_1m.yaml``
(seed/size/columns/filters/schema). Steps only consume the name — the sample
registry owns paths, filtering, and sampling.
"""

from pathlib import Path

import pandas as pd

from broadway.samples import read_named_sample
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
RESULTS = load_project_paths().results / HERE.name
SAMPLE_NAME = "fare_prediction_1m"
TARGET = "fare_amount"


def load_sample() -> pd.DataFrame:
    """Load the named fare_prediction_1m sample and print provenance."""
    sample = read_named_sample(SAMPLE_NAME)
    print(f"sample: {SAMPLE_NAME}@{sample.provenance['version']}")
    print(f"rows: {sample.provenance['row_count']}")
    print(f"artifact_sha256: {sample.provenance['artifact_sha256']}")
    return sample.df


def numeric_features(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Numeric candidate features, excluding the target."""
    return [c for c in df.select_dtypes("number").columns if c != target]


# Independent numeric features only.
# Excludes speed_mph (deterministic ratio of distance/time)
# Excludes location_ids (these are categorical zones, not continuous numbers)
INDEPENDENT_NUMERIC_FEATURES = [
    "trip_distance",
    "trip_duration_minutes",
]
