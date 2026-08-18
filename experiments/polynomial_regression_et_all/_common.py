"""Shared paths, constants, and dataset loaders for this experiment (no analysis logic).

The working-dataset binding (parquet path, filters, loaders, time_bucket) is
OWNED by `project.working` (single source of truth, config-driven) and
re-exported here so the step scripts keep their `from _common import ...`
imports unchanged. Only experiment-layout paths stay local.
"""

from pathlib import Path

from project.working import (
    MAX_DURATION_MINUTES,
    MIN_FARE,
    load_metered,
    load_working,
    time_bucket,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "experiments" / "results" / HERE.name
CLEAN_PARQUET = RESULTS / "sample50k.parquet"
