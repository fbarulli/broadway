"""Shared paths and plot helpers for the euromonitor series.

Follows the repo experiment convention (see fare_prediction/_common.py):
RESULTS = PATHS.results / HERE.name, and every step writes
`NN_name_describe.csv` + `NN_name.png` under it. The dataset lives at
project/data/euromonitor/dataset.csv (raw export from the Shiny app);
the series seed comes from project/config/experiments/euromonitor.yaml.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from broadway.utils import require_keys
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
DATA_PATH = PATHS.experiments.parent / "data" / "euromonitor" / "dataset.csv"


def load_euromonitor() -> pd.DataFrame:
    """Load the euromonitor dataset as raw strings (no silent coercion)."""
    return pd.read_csv(DATA_PATH, dtype=str)


def load_series_seed() -> int:
    """Shared seed from the euromonitor experiment config (YAML SSOT)."""
    path = PATHS.experiment_configs / "euromonitor.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require_keys(config, ["seed"], path.name)
    return int(config["seed"])


SEED = load_series_seed()
