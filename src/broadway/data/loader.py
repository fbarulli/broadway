"""Detect format (csv/parquet/excel) → load → optional lookup join."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.config.schema import DatasetContract, EnvironmentConfig
from broadway.data.join_audit import JoinAudit, audit_join

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


def load_with_audit(dataset: DatasetContract) -> tuple[pd.DataFrame, list[JoinAudit]]:
    path = Path(dataset.path)
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"unsupported format: {ext}")
    df = READERS[ext](path)
    audits: list[JoinAudit] = []
    for col, lookup in dataset.lookup_tables.items():
        right_on = lookup.key
        lookup_df = pd.read_csv(lookup.path)
        audits.append(audit_join(df, col, lookup, lookup_df))
        df = df.merge(lookup_df, left_on=col, right_on=right_on, how=MERGE_HOW, suffixes=("", "_lookup"))
    return df, audits
