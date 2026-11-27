"""Read CSV → infer dtypes, null counts → write configs/dataset/<name>.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from broadway.config.loader import CONFIGS_DIR, config_path
from broadway.config.schema import ColumnRole, ColumnSchema, DatasetContract, TaskType
from broadway.data.loader import ArtifactStore
from broadway.data.normalize import ColumnNormalizer
from broadway.discover.profile import DatasetProfile, build_profile
from broadway.discover.qq import QqOverview, plot_numeric_qq
from broadway.lineage.ids import node_id
from broadway.lineage.records import write_record
from broadway.reports.paths import FIGURES_DIR

logger = logging.getLogger(__name__)

DATASET_DIR = os.getenv("BROADWAY_DATASET_DIR", "dataset")
ARTIFACTS_DIR = os.getenv("BROADWAY_ARTIFACTS_DIR", "artifacts")
IDENTIFIER_THRESHOLD = float(os.getenv("BROADWAY_IDENTIFIER_THRESHOLD", "0.95"))


def _assign_role(col: str, target: str, dt_col: str | None, ignore: list[str]) -> ColumnRole:
    if col in ignore:
        return ColumnRole.IGNORE
    if col == target:
        return ColumnRole.TARGET
    if col == dt_col:
        return ColumnRole.DATETIME
    return ColumnRole.FEATURE


def _read(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path) if csv_path.endswith(".csv") else pd.read_parquet(csv_path)


def _build_contract(
    df: pd.DataFrame,
    csv_path: str,
    target: str,
    task: str,
    dt_col: str | None,
    ignore_cols: list[str],
    artifacts_subdir: str | None = None,
) -> DatasetContract:
    columns = {}
    for col in df.columns:
        role = _assign_role(col, target, dt_col, ignore_cols)
        columns[col] = ColumnSchema(
            dtype=("datetime64" if role == ColumnRole.DATETIME else str(df[col].dtype)),
            null_count=int(df[col].isna().sum()),
            role=role,
        )
    return DatasetContract(
        name=Path(csv_path).stem,
        path=csv_path,
        target=target,
        task=TaskType(task),
        datetime_column=dt_col,
        columns=columns,
        lookup_tables={},
        artifacts_subdir=artifacts_subdir,
    )


def _log_identifier_recommendations(contract: DatasetContract, profile: DatasetProfile) -> None:
    for col, col_profile in profile.columns.items():
        if contract.columns[col].role == ColumnRole.FEATURE and col_profile.identifier_score >= IDENTIFIER_THRESHOLD:
            logger.info(f"likely identifier: {col} (identifier_score={col_profile.identifier_score})")


def _discover_dir(contract: DatasetContract) -> Path:
    """The discover evidence dir for this dataset (dataset-scoped when declared)."""
    return ArtifactStore.from_contract(contract, Path(ARTIFACTS_DIR) / "discover").directory


def _qq_figures_dir(contract: DatasetContract) -> Path:
    """Tracked report figures for a shared dataset; its own store when scoped."""
    if contract.artifacts_subdir is None:
        return FIGURES_DIR
    return _discover_dir(contract) / "figures"


def _write_qq_overview(
    df: pd.DataFrame,
    source_path: str,
    contract: DatasetContract,
    exclude: list[str] | None = None,
) -> QqOverview:
    qq_dir = _discover_dir(contract)
    qq_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = _qq_figures_dir(contract)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return plot_numeric_qq(
        df, figures_dir, qq_dir / "qq_overview.json", source_path=source_path, exclude=exclude
    )


def run(
    csv: str,
    target: str,
    task: str,
    datetime_column: str | None = None,
    ignore_columns: list[str] | None = None,
    artifacts_subdir: str | None = None,
) -> None:
    ignore = ignore_columns or []
    df = _read(csv)
    contract = _build_contract(
        df, csv, target, task, datetime_column, ignore, artifacts_subdir
    )
    out_dir = CONFIGS_DIR / DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{contract.name}.yaml"
    logger.info(f"discover: writing {len(contract.columns)} columns to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(contract.model_dump(mode="json"), f, default_flow_style=False)

    profile = build_profile(contract.name, csv, df)
    profile_dir = _discover_dir(contract)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"discover: wrote {len(profile.columns)} column profiles to {profile_path}")
    _log_identifier_recommendations(contract, profile)
    _write_qq_overview(df, csv, contract)
    write_record(
        node_id("profile", contract.name),
        "profile",
        str(profile_path),
        [node_id("dataset", contract.name)],
    )


def profile(dataset_name: str) -> None:
    contract_path = config_path(f"{DATASET_DIR}/{dataset_name}.yaml")
    if not contract_path.exists():
        raise FileNotFoundError(f"dataset contract not found: {contract_path}")
    with open(contract_path, encoding="utf-8") as f:
        contract = DatasetContract(**yaml.safe_load(f))
    df = ColumnNormalizer.from_contract(contract).apply(_read(contract.path))
    result = build_profile(contract.name, contract.path, df)
    profile_dir = _discover_dir(contract)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.json"
    profile_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"profile: wrote {len(result.columns)} columns to {profile_path}")
    _write_qq_overview(
        df, contract.path, contract, exclude=contract.exclude_from_profiling
    )
    write_record(
        node_id("profile", contract.name),
        "profile",
        str(profile_path),
        [node_id("dataset", contract.name)],
    )
