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

import pandas as pd
import yaml

from broadway.utils import require_keys
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
DATA_PATH = PATHS.experiments.parent / "data" / "euromonitor" / "dataset.csv"


def load_column_mapping() -> dict:
    """Raw-export -> canonical column names (dataset contract YAML SSOT).

    The mapping lives ONCE in project/config/dataset/euromonitor.yaml
    (dataset.column_mapping) so the whole series renames columns project-wide
    through the shared loader instead of per-step.
    """
    path = PATHS.config / "dataset" / "euromonitor.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require_keys(config, ["dataset"], path.name)
    mapping = config["dataset"].get("column_mapping")
    if not mapping:
        raise ValueError(f"{path.name}: dataset.column_mapping is missing")
    return dict(mapping)


COLUMN_MAPPING = load_column_mapping()  # read once at import


def load_dataset() -> pd.DataFrame:
    """Load the ACTIVE dataset as raw strings (no silent coercion).

    THE dataset for this series until further notice: euromonitor. Rename
    DATA_PATH + this loader when the active dataset changes; steps import
    `load_dataset`, never a hardcoded path. Columns are canonicalized
    project-wide via COLUMN_MAPPING (config/dataset/euromonitor.yaml).
    """
    df = pd.read_csv(DATA_PATH, dtype=str)
    return df.rename(columns=COLUMN_MAPPING)


def load_dataset_deduped() -> pd.DataFrame:
    """The DEDUPED dataset (06 tiered dedupe) — the matching-stage input.

    Step 03+ matching consumes this (one row per retailer-product after
    marketplace-listing collapse); the raw export remains the source of
    truth via load_dataset. Columns are already canonical (written by 06).
    """
    path = DATA_PATH.with_name("dataset_deduped.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run 06_dedupe.py first")
    return pd.read_csv(path, dtype=str)


# Backward-compatible alias for existing steps; prefer load_dataset going
# forward (single active-dataset entry point).
load_euromonitor = load_dataset


def load_series_seed() -> int:
    """Shared seed from the euromonitor experiment config (YAML SSOT)."""
    path = PATHS.experiment_configs / "euromonitor.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require_keys(config, ["seed"], path.name)
    return int(config["seed"])


SEED = load_series_seed()
