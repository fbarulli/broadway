"""Shared paths and plot helpers for the euromonitor series.

Follows the repo experiment convention for a series' shared paths and plot helpers:
RESULTS = PATHS.results / HERE.name, and every step writes
`NN_name_describe.csv` + `NN_name.png` under it. The dataset lives at
project/data/euromonitor/dataset.csv (raw export from the Shiny app);
the series seed comes from project/config/experiments/euromonitor.yaml.
"""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import pandas as pd
import yaml
from _text import extract_volume_ml

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


# ---------------------------------------------------------------------------
# Shared dataframe/regex helpers used by multiple steps.
# ---------------------------------------------------------------------------

def has_barcode(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: row has a non-empty barcode (GTIN)."""
    return df["barcode"].fillna("").astype(str).str.len() > 0


def multi_retailer_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: known barcode that appears under more than one retailer."""
    return has_barcode(df) & (
        df.groupby("barcode")["retailer"].transform("nunique") > 1
    )


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: non-null count, cardinality, numeric_like, stored dtype.

    numeric_like is the fraction of the first 5k non-null values that parse as
    numbers (loader reads dtype=str, so this shows the real content signal).
    Single source for 01's dtype table and 01b's column scatter.
    """
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        cardinality = int(df[col].dropna().nunique())
        numeric_like = 0.0
        if non_null:
            sample = df[col].dropna().head(5000)
            numeric_like = round(
                float(pd.to_numeric(sample, errors="coerce").notna().mean()), 4)
        rows.append({
            "column": col,
            "stored_dtype": str(df[col].dtype),
            "non_null": non_null,
            "cardinality": cardinality,
            "numeric_like": numeric_like,
        })
    return pd.DataFrame(rows)


def canonical_volume(series: pd.Series) -> pd.DataFrame:
    """Title series -> (canonical_volume_ml, canonical_volume_ambiguous) frame.

    Canonical volume comes from title ONLY (extract_volume_ml); the ambiguous
    flag marks bare oz/ounce (weight vs fluid). Single source for the
    extract-volume projection used by 01e/01f/01h/02/02b/02c.
    """
    vol = series.map(extract_volume_ml)
    return pd.DataFrame({
        "canonical_volume_ml": vol.map(lambda t: t[0]),
        "canonical_volume_ambiguous": vol.map(lambda t: t[1]),
    })


def parse_list_cell(value):
    """Parse a list serialized as a Python literal in a CSV cell, with fallback.

    02b/03 store Python list reprs in cells (sample_names, canonical_volumes).
    Returns the parsed list, [value] for a scalar, [] for a missing cell, and
    survives a malformed repr by returning [value] instead of crashing.
    """
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if pd.isna(value):  # None / NaN / NaT cell
        return []
    return [value]


def barcode_agreement_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    *,
    sample: bool = False,
) -> pd.DataFrame:
    """Per-barcode volume-agreement table over multi-retailer barcode groups.

    `columns` is a list of (column_name, label) pairs. For each pair the result
    has `{label}_volumes` (sorted unique non-null values) and `{label}_agree`
    (True when the group's detected values collapse to one unique value, None
    when none are detected). Empty groups are NOT dropped here — callers
    dropna() the column they validate (the honest denominator: empty groups are
    excluded, never counted as trivially agreeing). `sample=True` adds
    sample_names/sample_retailers (first 5 unique).
    """
    multi = df[multi_retailer_mask(df)]

    def _agg(x: pd.DataFrame) -> pd.Series:
        row: dict = {"retailers": x["retailer"].nunique(), "skus": len(x)}
        for col, label in columns:
            vols = sorted(x[col].dropna().unique().tolist())
            row[f"{label}_volumes"] = vols
            row[f"{label}_agree"] = (len(vols) <= 1) if vols else None
        if sample:
            row["sample_names"] = x["title"].dropna().unique().tolist()[:5]
            row["sample_retailers"] = x["retailer"].unique().tolist()[:5]
        return pd.Series(row)

    return multi.groupby("barcode").apply(_agg, include_groups=False).reset_index()
