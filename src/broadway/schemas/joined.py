"""Joined-loader schema: raw contract columns plus lookup columns under the
loader's own merge naming (Decision 6 — derived, never a second copy of the
suffix rule)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from broadway.config.schema import DatasetContract
from broadway.data.loader import merged_lookup_column_names


def _read_lookup_columns(path: str) -> list[str]:
    frame_path = Path(path)
    if not frame_path.exists():
        raise FileNotFoundError(f"lookup table not found: {frame_path}")
    return list(pd.read_csv(frame_path, nrows=0).columns)


def joined_schema_columns(dataset: DatasetContract) -> frozenset[str]:
    """Post-join column set: raw columns plus each lookup's columns renamed by
    ``merged_lookup_column_names``, accumulated in ``lookup_tables`` order."""
    columns: set[str] = set(dataset.columns)
    for lookup in dataset.lookup_tables.values():
        names = merged_lookup_column_names(columns, _read_lookup_columns(lookup.path))
        columns |= set(names.values())
    return frozenset(columns)
