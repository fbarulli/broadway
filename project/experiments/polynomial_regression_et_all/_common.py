"""Shared paths, constants, and dataset loaders for this experiment (no analysis logic).

The working-dataset binding (parquet path, filters, loaders, time_bucket) is
OWNED by `project.working` (single source of truth, config-driven) and
re-exported here so the step scripts keep their `from _common import ...`
imports unchanged. Only experiment-layout paths stay local.
"""

from pathlib import Path

from project.paths import load_project_paths
from project.working import (
    MAX_DURATION_MINUTES,
    MIN_TARGET_VALUE,
    load_metered,
    load_working,
    time_bucket,
)

__all__ = (
    "CLEAN_PARQUET",
    "MAX_DURATION_MINUTES",
    "MIN_TARGET_VALUE",
    "RESULTS",
    "load_metered",
    "load_working",
    "time_bucket",
)

HERE = Path(__file__).resolve().parent
RESULTS = load_project_paths().results / HERE.name
CLEAN_PARQUET = RESULTS / "sample50k.parquet"
