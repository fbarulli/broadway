"""Read CSV → infer dtypes, null counts → write configs/dataset/<name>.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from broadway.config.loader import CONFIGS_DIR
from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract, TaskType

logger = logging.getLogger(__name__)

DATASET_DIR = os.getenv("BROADWAY_DATASET_DIR", "dataset")


def _assign_role(col: str, target: str, dt_col: str | None, ignore: list[str]) -> ColumnRole:
    if col in ignore:
        return ColumnRole.IGNORE
    if col == target:
        return ColumnRole.TARGET
    if col == dt_col:
        return ColumnRole.DATETIME
    return ColumnRole.FEATURE


def _build_contract(
    csv_path: str, target: str, task: str, dt_col: str | None, ignore_cols: list[str],
) -> DatasetContract:
    df = pd.read_csv(csv_path)
    columns = {
        col: ColumnSchema(
            dtype=str(df[col].dtype),
            null_count=int(df[col].isna().sum()),
            role=_assign_role(col, target, dt_col, ignore_cols),
        )
        for col in df.columns
    }
    return DatasetContract(
        name=Path(csv_path).stem,
        path=csv_path,
        target=target,
        task=TaskType(task),
        datetime_column=dt_col,
        columns=columns,
        lookup_tables={},
        row_count=len(df),
    )


def run(
    csv: str,
    target: str,
    task: str,
    datetime_column: str | None = None,
    ignore_columns: list[str] | None = None,
) -> None:
    ignore = ignore_columns or []
    contract = _build_contract(csv, target, task, datetime_column, ignore)
    out_dir = CONFIGS_DIR / DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{contract.name}.yaml"
    logger.info(f"discover: writing {len(contract.columns)} columns to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(contract.model_dump(mode="json"), f, default_flow_style=False)
