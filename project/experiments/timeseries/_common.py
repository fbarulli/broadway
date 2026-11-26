"""Shared paths, constants, and loaders for timeseries experiments.

Continuity rule: series work needs CONTIGUOUS time coverage. The pinned
``fare_prediction_1m`` sample is a RANDOM draw — fine for cross-sectional
modeling, wrong for series work (bucket counts stay unbiased but small
buckets get noisy and gap analysis is meaningless). The default source here
is therefore the cleaned canonical frame, not the sample.
"""

from pathlib import Path

import pandas as pd
import yaml

from broadway.reports.decision import DecisionRule, DecisionRuleSet
from broadway.utils import require_keys
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
EXPERIMENT_CONFIG = PATHS.experiment_configs / "timeseries.yaml"
DECISION_RULES = DecisionRuleSet.from_yaml(EXPERIMENT_CONFIG)


def _load_experiment_seed() -> int:
    """Shared seed from the timeseries experiment config (YAML SSOT)."""
    config = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    require_keys(config, ["seed"], EXPERIMENT_CONFIG.name)
    return int(config["seed"])


SEED = _load_experiment_seed()


def load_decision_rule(step: str) -> DecisionRule:
    """The step's decision rule from the timeseries experiment config."""
    return DECISION_RULES.rule(step)

CANONICAL = Path("data/processed/taxi_canonical.parquet")
RAW_MONTHS = {
    "2024-01": Path("data/raw/yellow_tripdata_2024-01.parquet"),
    "2024-02": Path("data/raw/yellow_tripdata_2024-02.parquet"),
    "2024-03": Path("data/raw/yellow_tripdata_2024-03.parquet"),
}
DATETIME_COL = "pickup_datetime"


def load_canonical(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned canonical frame (Jan-Mar 2024, contiguous)."""
    if not CANONICAL.exists():
        raise FileNotFoundError(f"canonical frame not found: {CANONICAL}")
    df = pd.read_parquet(CANONICAL, columns=columns)
    print(f"canonical: {len(df)} rows from {CANONICAL}")
    return df


def load_raw_month(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one raw TLC month (provenance checks; uncleaned)."""
    if month not in RAW_MONTHS:
        raise ValueError(f"unknown month {month!r} (known: {sorted(RAW_MONTHS)})")
    path = RAW_MONTHS[month]
    if not path.exists():
        raise FileNotFoundError(f"raw month not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    print(f"raw {month}: {len(df)} rows from {path}")
    return df
