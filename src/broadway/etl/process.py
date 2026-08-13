import logging
import os
from pathlib import Path

import pandas as pd

from broadway.etl import process_config as cfg
from broadway.features.contracts import validate_raw_schema
from broadway.features.schema import RAW_FEATURES, TARGET

logger = logging.getLogger(__name__)


def get_raw_files() -> list[Path]:
    raw_data_dir = Path(cfg.raw_dir)
    files = sorted(raw_data_dir.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {raw_data_dir}")
    return files


def read_raw_data(files: list[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        logger.info(f"Reading {f.name}...")
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
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


def compute_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    return df


def filter_valid_duration(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[
        (df[TARGET] >= cfg.min_trip_duration_minutes) &
        (df[TARGET] <= cfg.max_trip_duration_minutes)
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


def select_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[RAW_FEATURES + [TARGET]].dropna()
    logger.info(f"After dropna: {len(df)} ({n_before - len(df)} dropped)")
    return df


def save_processed_data(df: pd.DataFrame) -> None:
    processed_dir = Path(cfg.processed_dir)
    processed_file = processed_dir / cfg.processed_file
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file} ({len(df)} rows)")


def process_data() -> None:
    logger.info("Processing data...")

    files = get_raw_files()
    df = read_raw_data(files)
    df = sample_for_ci(df)

    logger.info("Filtering invalid trips...")
    df = filter_valid_trips(df)
    df = compute_trip_duration(df)
    df = filter_valid_duration(df)
    df = filter_valid_passenger_count(df)

    df = rename_columns(df)
    df = select_and_clean_columns(df)

    validate_raw_schema(df)

    save_processed_data(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    process_data()
