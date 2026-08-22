"""Detect format (csv/parquet/excel) → load → optional lookup join."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from broadway.config.schema import DatasetContract, EnvironmentConfig
from broadway.data.join_audit import JoinAudit, audit_join
from broadway.data.lookup_value_audit import LookupValueAudit, audit_lookup_values

READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}

MERGE_HOW = "left"


def canonical_path(dataset: DatasetContract, environment: EnvironmentConfig) -> Path:
    return (
        Path(environment.data_dir)
        / environment.processed_subdir
        / f"{dataset.name}_canonical.parquet"
    )


def load(dataset: DatasetContract) -> pd.DataFrame:
    return load_with_audit(dataset)[0]


def load_with_audit(dataset: DatasetContract) -> tuple[pd.DataFrame, list[JoinAudit], list[LookupValueAudit]]:
    path = Path(dataset.path)
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"unsupported format: {ext}")
    df = READERS[ext](path)
    audits: list[JoinAudit] = []
    value_audits: list[LookupValueAudit] = []
    for col, lookup in dataset.lookup_tables.items():
        right_on = lookup.key
        lookup_df = pd.read_csv(lookup.path, keep_default_na=False, na_values=lookup.na_values)
        audit = audit_join(df, col, lookup, lookup_df)
        audits.append(audit)
        merged_names = {c: (c if c not in df.columns else c + "_lookup") for c in lookup_df.columns}
        df = df.merge(lookup_df, left_on=col, right_on=right_on, how=MERGE_HOW, suffixes=("", "_lookup"))
        value_audits.append(
            audit_lookup_values(
                df_merged=df,
                left_key=col,
                lookup=lookup,
                lookup_df=lookup_df,
                merged_names=merged_names,
                matched=audit.matched,
            )
        )
    return df, audits, value_audits


def read_sample(
    dataset: DatasetContract,
    sample: int | None = None,
    seed: int | None = None,
    columns: list[str] | None = None,
    *,
    full: bool = False,
) -> pd.DataFrame:
    """Seeded random sample of ``dataset.path`` via a lazy scan.

    Draws directly from the dataset's raw parquet — NOT the dev/live mode
    caches. Optional ``columns`` prunes to those columns only. ``sample=None``
    requires ``full=True`` (loading the full dataset is deliberate). ``seed`` is
    passed through unchanged; callers own reproducibility.
    """
    if sample is None and not full:
        raise ValueError("pass sample=<n>, or full=True to load the full dataset")
    lf = pl.scan_parquet(dataset.path)
    if columns:
        lf = lf.select(columns)
    df = lf.collect()
    if sample is not None:
        df = df.sample(n=sample, seed=seed)
    return df.to_pandas()
