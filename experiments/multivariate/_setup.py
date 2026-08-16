"""Shared setup for the multivariate experiment (categorical breakdown).

The working dataset loader, its filter constants, and `time_bucket` are OWNED
by `project.working` (single source of truth, config-driven) and imported
directly here — no importlib, no shadowing. The zone-lookup path/columns are
owned by `project.data`. Analysis policy lives in
`configs/experiments/multivariate.yaml` (config files only, never hardcoded).
This module is deliberately NOT named `_common` because the univariate modules
import `_common` by name; a same-named module here would shadow them.
"""

from pathlib import Path

import pandas as pd
import yaml

from broadway.utils import require_keys
from project.data import LOOKUP_PATH, ZONE_BOROUGH_COL, ZONE_ID_COL
from project.working import WORKING_DATASET, load_metered, time_bucket

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results" / "multivariate"
REPO = HERE.parents[1]

CONFIG_KEYS = ["target", "value_counts_head", "sample_role", "categorical",
               "borough", "sample", "borough_dummies", "labels", "baseline",
               "geography_premium", "model_verdicts"]


def load_config() -> dict:
    """Analysis policy from configs/experiments/multivariate.yaml (never hardcoded)."""
    cfg_path = REPO / "configs" / "experiments" / "multivariate.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    require_keys(config, CONFIG_KEYS, cfg_path.name)
    return config


def load_zones() -> pd.DataFrame:
    """LocationID -> Borough lookup (path/columns owned by project.data)."""
    return pd.read_csv(LOOKUP_PATH, usecols=[ZONE_ID_COL, ZONE_BOROUGH_COL])


def add_boroughs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join pickup + dropoff boroughs; fail loudly on broken joins."""
    zones = load_zones()
    out = df
    for key in ("pickup", "dropoff"):
        spec = cfg["borough"][key]
        if spec["join_on"] not in out.columns:
            raise ValueError(f"borough join: column '{spec['join_on']}' not in data")
        before = len(out)
        out = out.merge(zones, left_on=spec["join_on"], right_on=ZONE_ID_COL,
                        how="left").rename(
                            columns={ZONE_BOROUGH_COL: spec["column"]})
        if len(out) != before:
            raise ValueError(
                f"borough join on '{spec['join_on']}' changed row count "
                f"({before} -> {len(out)}) — duplicate LocationIDs in lookup?")
        unmapped = int(out[spec["column"]].isna().sum())
        if unmapped:
            print(f"borough join '{spec['column']}': {unmapped} rows unmapped "
                  f"(kept as NaN — callers must handle or drop them)")
    return out


def load_manhattan_sample(cfg: dict) -> pd.DataFrame:
    """Metered rows restricted to the config pickup borough (the 'manhattan_sample')."""
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    df = add_boroughs(df, cfg)
    pickup_col = cfg["borough"]["pickup"]["column"]
    keep = cfg["sample"]["pickup_borough"]
    total = len(df)
    sample = df[df[pickup_col] == keep]
    print(f"sample filter '{pickup_col}' == '{keep}': {len(sample)} of {total} "
          f"({len(sample) / total:.1%})")
    if sample.empty:
        raise ValueError(f"sample filter yielded 0 rows "
                         f"(pickup borough '{keep}' absent from data)")
    return sample


def build_borough_dummies(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One-hot dummies for the config borough column (config reference dropped).

    Wraps the Categorical in a Series so get_dummies keeps df's index — a
    bare Categorical would emit a RangeIndex and silently misalign a concat.
    Callers decide missing-value handling (drop or fill) before calling.
    """
    spec = cfg["borough_dummies"]
    col = spec["column"]
    categories = [spec["reference"]] + sorted(
        v for v in df[col].dropna().unique() if v != spec["reference"])
    coded = pd.Series(pd.Categorical(df[col], categories=categories),
                      index=df.index)
    return pd.get_dummies(coded, prefix="borough",
                          drop_first=True, dtype=float)


def load_stratified_sample(cfg: dict, per_borough: int) -> pd.DataFrame:
    """Borough-stratified sample of the metered data (population B)."""
    df = load_metered()
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["time_bucket"] = df["pickup_hour"].map(time_bucket)
    df = add_boroughs(df, cfg)
    col = cfg["borough"]["pickup"]["column"]
    seed = cfg["baseline"]["seed"]
    groups = [group.sample(n=min(per_borough, len(group)), random_state=seed)
              for _, group in df.groupby(col)]
    out = pd.concat(groups).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"stratified sample ({col}): {len(out)} rows")
    return out


def load_baseline_sample(cfg: dict, spec: dict) -> pd.DataFrame:
    """Load a baseline population's sample (source from the population spec)."""
    source = spec["source"]
    if source == "manhattan":
        return load_manhattan_sample(cfg)
    if source == "stratified":
        return load_stratified_sample(cfg, spec["per_borough"])
    raise ValueError(f"unknown baseline source '{source}' "
                     "(expected 'manhattan' or 'stratified')")
