"""Dataset-specific loaders and constants for the NYC taxi project."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from broadway.config.schema import (
    DatasetContract,
    FeaturesStep,
    StatsStep,
    TrainStep,
)
from broadway.contracts.pandera import build_raw_schema
from broadway.stats.effect_size import group_imbalance

logger = logging.getLogger(__name__)

_DATASET_YAML = Path("configs/dataset/taxi.yaml")
_contract = DatasetContract(**yaml.safe_load(_DATASET_YAML.read_text()))

_STATS_YAML = Path("configs/step/stats.yaml")
_stats = StatsStep(**yaml.safe_load(_STATS_YAML.read_text()))

_TRAIN_YAML = Path("configs/step/train.yaml")
_train = TrainStep(**yaml.safe_load(_TRAIN_YAML.read_text()))

_FEATURES_YAML = Path("configs/step/features.yaml")
_features = FeaturesStep(**yaml.safe_load(_FEATURES_YAML.read_text()))


def _lookup_path(contract: DatasetContract) -> Path:
    return Path(next(iter(contract.lookup_tables.values())).path)


DATA_PATH = Path(_contract.path)
LOOKUP_PATH = Path(_lookup_path(_contract))
BOROUGHS = _stats.group_values
MIN_ROWS_FOR_SAMPLING = _stats.min_rows_for_sampling
SAMPLE_FRACTION = _stats.per_group_sample_fraction
TIME_SPLIT_CUTOFF = _stats.time_split_cutoff
ACF_LAGS = _stats.acf_lags

FEATURE_LOOKUP_PATH = _features.lookup_path
FEATURE_ENCODING_SMOOTHING = _features.encoding_smoothing
FEATURE_FREQUENCY_FILL = _features.frequency_fill
FEATURE_RUSH_HOUR_MORNING_START = _features.rush_hour_morning_start
FEATURE_RUSH_HOUR_MORNING_END = _features.rush_hour_morning_end
FEATURE_RUSH_HOUR_EVENING_START = _features.rush_hour_evening_start
FEATURE_RUSH_HOUR_EVENING_END = _features.rush_hour_evening_end
FEATURE_NIGHT_START = _features.night_start
FEATURE_NIGHT_END = _features.night_end
FEATURE_PASSENGER_COUNT_MIN = _features.passenger_count_min
FEATURE_PASSENGER_COUNT_MAX = _features.passenger_count_max

def _resolve_mode(mode: str | None = None) -> str:
    m = mode or os.getenv("DATA_MODE", "dev")
    if m not in ("dev", "live"):
        raise ValueError(f"DATA_MODE must be 'dev' or 'live', got {m!r}")
    return m


def _sample_size(mode: str) -> int:
    return getattr(_stats, f"sample_size_{mode}")


def _time_slice_bounds(mode: str) -> tuple[str, str]:
    return (
        getattr(_stats, f"time_slice_start_{mode}"),
        getattr(_stats, f"time_slice_end_{mode}"),
    )


def _cache_path(mode: str) -> Path:
    return RESULTS_DIR / f"joined_sample_{mode}.parquet"


def _meta_path(mode: str) -> Path:
    return RESULTS_DIR / f"sample_meta_{mode}.json"


MODE = _resolve_mode()
SAMPLE_SIZE = _sample_size(MODE)
TIME_SLICE_START, TIME_SLICE_END = _time_slice_bounds(MODE)

RANDOM_STATE = _train.random_state
N_ESTIMATORS = _train.n_estimators
LEARNING_RATE = _train.learning_rate
NUM_LEAVES = _train.num_leaves
SUBSAMPLE = _train.subsample
COLSAMPLE_BYTREE = _train.colsample_bytree
QUANTILE_TAIL = _train.quantile_tail

ZONE_ID_COL = "LocationID"
ZONE_BOROUGH_COL = "Borough"
PICKUP_LOCATION_COL = next(k for k in _contract.lookup_tables if "pickup" in k)
DROPOFF_LOCATION_COL = next(k for k in _contract.lookup_tables if "dropoff" in k)
DATETIME_COL = _contract.datetime_column
TRIP_DISTANCE_COL = "trip_distance"
PASSENGER_COUNT_COL = "passenger_count"
TARGET_COL = _contract.target

PICKUP_BOROUGH_COL = "pickup_borough"
DURATION_COL = "trip_duration_minutes"

RESULTS_DIR = Path("results")


def _quality_report_path() -> Path:
    return RESULTS_DIR / "quality_report.json"

_BATCH_SIZE = 100_000


def _params_hash(mode: str | None = None) -> str:
    m = _resolve_mode(mode)
    payload = {
        "mode": m,
        "sample_size": _sample_size(m),
        "random_state": RANDOM_STATE,
        "group_col": PICKUP_BOROUGH_COL,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _load_zones() -> pd.DataFrame:
    return pd.read_csv(LOOKUP_PATH, usecols=[ZONE_ID_COL, ZONE_BOROUGH_COL])


def _join_boroughs(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        _load_zones(),
        left_on=PICKUP_LOCATION_COL,
        right_on=ZONE_ID_COL,
        how="left",
    ).rename(columns={ZONE_BOROUGH_COL: PICKUP_BOROUGH_COL})


def read_training_data(
    columns: list[str] | None = None, filters: list | None = None
) -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH, columns=columns, filters=filters)
    return build_raw_schema(_contract).validate(raw)


def load_boroughs_pandas() -> pd.DataFrame:
    return _join_boroughs(read_training_data())


def load_stratified_sample(mode: str | None = None) -> pd.DataFrame:
    m = _resolve_mode(mode)
    current_hash = _params_hash(m)
    cache_path = _cache_path(m)
    meta_path = _meta_path(m)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Call generate_sample_cache() first."
        )

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("params_hash") != current_hash:
            logger.warning(
                "sample params changed (current=%s, cached=%s). "
                "Run generate_sample_cache() to regenerate.",
                current_hash,
                meta.get("params_hash"),
            )

    return pd.read_parquet(cache_path)


def generate_sample_cache(mode: str | None = None) -> None:
    m = _resolve_mode(mode)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(DATA_PATH)
    frac = min(1.0, _sample_size(m) / pf.metadata.num_rows)

    counts, sums = _stream_counts(pf)
    targets = {
        borough: n if n <= MIN_ROWS_FOR_SAMPLING else round(frac * n)
        for borough, n in counts.items()
    }

    cache_path = _cache_path(m)
    meta_path = _meta_path(m)
    sample = _stream_sample(pf, targets)
    sample.to_parquet(cache_path)
    meta_path.write_text(json.dumps({"params_hash": _params_hash(m)}))
    logger.info("wrote %d rows to %s", len(sample), cache_path)

    _write_quality_report(
        counts, {borough: total / counts[borough] for borough, total in sums.items()}
    )


def _stream_counts(
    pf: pq.ParquetFile,
) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    for df in _iter_joined_batches(pf):
        grouped = df.groupby(PICKUP_BOROUGH_COL)[DURATION_COL].agg(["count", "sum"])
        for borough, row in grouped.iterrows():
            counts[borough] = counts.get(borough, 0) + int(row["count"])
            sums[borough] = sums.get(borough, 0.0) + float(row["sum"])
    return counts, sums


def _stream_sample(pf: pq.ParquetFile, targets: dict[str, int]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir: dict[str, pd.DataFrame] = {}
    seen: dict[str, int] = {}

    for df in _iter_joined_batches(pf):
        for borough, group in df.groupby(PICKUP_BOROUGH_COL):
            target = targets.get(borough, 0)
            if target <= 0:
                continue

            old = reservoir.get(borough)
            seen_before = seen.get(borough, 0)
            n_new = len(group)
            n_keep_total = min(target, seen_before + n_new)
            n_drawn_new = int(rng.hypergeometric(n_new, seen_before, n_keep_total))
            n_keep_old = n_keep_total - n_drawn_new

            parts: list[pd.DataFrame] = []
            if old is not None and n_keep_old > 0:
                parts.append(old.sample(n=n_keep_old, random_state=rng))
            if n_drawn_new > 0:
                parts.append(group.sample(n=n_drawn_new, random_state=rng))

            reservoir[borough] = pd.concat(parts, ignore_index=True) if parts else old
            seen[borough] = seen_before + n_new

    return pd.concat(list(reservoir.values()), ignore_index=True)


def _iter_joined_batches(pf: pq.ParquetFile) -> Iterator[pd.DataFrame]:
    for batch in pf.iter_batches(batch_size=_BATCH_SIZE):
        yield _join_boroughs(batch.to_pandas())


def load_borough_durations(mode: str | None = None) -> dict[str, np.ndarray]:
    df = load_stratified_sample(mode)
    return {
        borough: df.loc[
            df[PICKUP_BOROUGH_COL] == borough, DURATION_COL
        ].to_numpy().astype(float)
        for borough in BOROUGHS
    }


def load_time_slice(mode: str | None = None) -> pd.DataFrame:
    start, end = _time_slice_bounds(_resolve_mode(mode))
    df = read_training_data(
        filters=[
            (DATETIME_COL, ">=", pd.Timestamp(start)),
            (DATETIME_COL, "<", pd.Timestamp(end)),
        ]
    )
    return _join_boroughs(df).sort_values(DATETIME_COL).reset_index(drop=True)


def inspect_schema() -> None:
    pf = pq.ParquetFile(DATA_PATH)
    schema = pf.schema_arrow

    print("=== Schema ===")
    for field in schema:
        print(f"  {field.name}: {field.type}")

    print("=== Row count ===")
    print(pf.metadata.num_rows)

    print("=== Sample rows ===")
    print(next(pf.iter_batches(batch_size=10)).to_pandas())

    print("=== Column names ===")
    print(schema.names)


def write_quality_report() -> None:
    if _quality_report_path().exists():
        return

    df = load_boroughs_pandas()
    group_sizes = df.groupby(PICKUP_BOROUGH_COL).size().to_dict()
    group_means = df.groupby(PICKUP_BOROUGH_COL)[DURATION_COL].mean().to_dict()
    _write_quality_report(group_sizes, group_means)


def _write_quality_report(
    group_sizes: dict[str, int], group_means: dict[str, float]
) -> None:
    sizes = {k: int(v) for k, v in group_sizes.items()}
    means = {k: round(float(v), 2) for k, v in group_means.items()}
    total_n = sum(sizes.values())

    report = {
        "group_sizes": sizes,
        "group_means": means,
        "total_n": int(total_n),
        "imbalance_ratio": float(group_imbalance(sizes)),
        "any_small_group": any(
            size <= MIN_ROWS_FOR_SAMPLING for size in sizes.values()
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _quality_report_path().write_text(json.dumps(report, indent=2))
