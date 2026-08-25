import logging
import os
import re
from pathlib import Path

import pandera as pa
import pandas as pd
import polars as pl
import yaml

from broadway.config.loader import CONFIGS_DIR
from broadway.config.schema import DatasetContract
from broadway.contracts.pandera import build_raw_schema, pandera_dtype
from project.etl import process_config as cfg
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import write_record

logger = logging.getLogger(__name__)

# Drop-reason grammar shared with broadway.etl.module._explained_rows:
# reasons must end in "-<N> rows" so explained drops stay machine-parseable.
_ROW_DELTA = re.compile(r"-(\d+) rows$")

StageLedger = list[tuple[str, int]]


def get_raw_files() -> list[Path]:
    raw_data_dir = Path(cfg.raw_dir)
    files = sorted(raw_data_dir.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {raw_data_dir}")
    return files


def read_raw_data(files: list[Path]) -> pd.DataFrame:
    lf = pl.scan_parquet([str(f) for f in files])
    df = lf.collect().to_pandas()
    logger.info(f"Combined raw rows: {len(df)}")
    return df


def sample_for_ci(df: pd.DataFrame) -> pd.DataFrame:
    if os.getenv("CI") == "true":
        logger.info(f"CI mode detected: sampling {cfg.ci_sample_size:,} raw rows for fast processing.")
        if len(df) > cfg.ci_sample_size:
            df = df.sample(n=cfg.ci_sample_size, random_state=cfg.random_state)
    return df


def filter_valid_trips(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[
        (df["trip_distance"] > cfg.min_trip_distance) &
        (df["trip_distance"] < cfg.max_trip_distance) &
        (df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]) &
        (df["tpep_pickup_datetime"] >= pd.to_datetime(cfg.min_pickup_datetime))
    ].copy()
    logger.info(f"After distance/time filters: {len(df)} ({n_before - len(df)} dropped)")
    return df


def compute_trip_duration(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df[target] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return df


def filter_valid_duration(df: pd.DataFrame, target: str) -> pd.DataFrame:
    n_before = len(df)
    df = df[
        (df[target] >= cfg.min_trip_duration_minutes) &
        (df[target] <= cfg.max_trip_duration_minutes)
    ].copy()
    logger.info(f"After duration filter: {len(df)} ({n_before - len(df)} dropped)")
    return df


def filter_valid_passenger_count(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    pc = df["passenger_count"]
    valid = (
        pc.notna()
        & (pc >= cfg.min_passenger_count)
        & (pc <= cfg.max_passenger_count)
        & (pc == pc.round())
    )
    df = df[valid].copy()
    logger.info(f"After passenger_count filter: {len(df)} ({n_before - len(df)} dropped)")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=cfg.rename_map)


def select_and_clean_columns(
    df: pd.DataFrame,
    contract: DatasetContract,
    ledger: StageLedger | None = None,
) -> pd.DataFrame:
    cols = list(contract.columns.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"raw data missing required contract column(s): {sorted(missing)}")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        logger.info(f"dropping extra column(s) not in contract: {sorted(extra)}")
    n_before = len(df)
    df = df[cols].dropna()
    logger.info(f"After dropna: {len(df)} ({n_before - len(df)} dropped)")
    if ledger is not None:
        ledger.append(("dropna", len(df)))
    n_duplicates_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"After drop_duplicates: {len(df)} ({n_duplicates_before - len(df)} dropped)")
    if ledger is not None:
        ledger.append(("duplicates", len(df)))
    return df


def validate_contract_schema(df: pd.DataFrame, contract: DatasetContract) -> None:
    for name, col in contract.columns.items():
        if not col.dtype.strip():
            raise ValueError(f"column '{name}' is missing a dtype in the dataset contract")
        if pandera_dtype(col.dtype) is pa.Object:
            raise ValueError(f"column '{name}' has unsupported dtype '{col.dtype}'")
    build_raw_schema(contract).validate(df)


def save_processed_data(df: pd.DataFrame) -> None:
    processed_dir = Path(cfg.processed_dir)
    processed_file = processed_dir / cfg.processed_file
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file} ({len(df)} rows)")


def _load_contract(dataset: str) -> DatasetContract:
    path = CONFIGS_DIR / "dataset" / f"{dataset}.yaml"
    return DatasetContract(**yaml.safe_load(path.read_text()))


def _stage_ledger(
    df: pd.DataFrame, contract: DatasetContract
) -> tuple[pd.DataFrame, StageLedger]:
    """Run the ingest filter chain, recording (stage, rows_out) after each stage."""
    stages: StageLedger = []
    n_before = len(df)
    df = sample_for_ci(df)
    if len(df) < n_before:
        stages.append(("ci_sample", len(df)))
    logger.info("Filtering invalid trips...")
    df = filter_valid_trips(df)
    stages.append(("filter_valid_trips", len(df)))
    df = compute_trip_duration(df, contract.target)
    stages.append(("compute_trip_duration", len(df)))
    df = filter_valid_duration(df, contract.target)
    stages.append(("filter_valid_duration", len(df)))
    df = filter_valid_passenger_count(df)
    stages.append(("filter_valid_passenger_count", len(df)))
    df = rename_columns(df)
    df = select_and_clean_columns(df, contract, stages)
    return df, stages


def _ingest_audit(
    rows_in: int,
    stages: StageLedger,
    columns_before: list[str],
    columns_after: list[str],
) -> TransformAudit:
    rows_out = stages[-1][1] if stages else rows_in
    dropped_total = max(0, rows_in - rows_out)
    reasons: list[str] = []
    previous = rows_in
    for name, stage_rows in stages:
        dropped = previous - stage_rows
        if dropped > 0:
            reasons.append(f"{name}: -{dropped} rows")
        previous = stage_rows
    explained = sum(int(m.group(1)) for r in reasons if (m := _ROW_DELTA.search(r)))
    return TransformAudit(
        rows_in=rows_in, rows_out=rows_out, reasons=reasons,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=max(0, dropped_total - explained),
        columns_before=columns_before, columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )


def process_data(dataset: str) -> None:
    logger.info("Processing data...")

    contract = _load_contract(dataset)

    files = get_raw_files()
    df = read_raw_data(files)
    rows_in = len(df)
    columns_before = list(df.columns)

    df, stages = _stage_ledger(df, contract)

    validate_contract_schema(df, contract)

    save_processed_data(df)

    processed_path = Path(cfg.processed_dir) / cfg.processed_file
    write_record(
        node_id("ingest", dataset),
        "ingest",
        str(processed_path),
        [node_id("dataset", dataset)],
        audit=_ingest_audit(rows_in, stages, columns_before, list(df.columns)),
    )
    logger.info("ingest lineage record written")
