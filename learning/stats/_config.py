import json
import hashlib
import logging
import yaml
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from broadway.config.schema import FeaturesStep, StatsStep, TrainStep

logger = logging.getLogger(__name__)

_STATS_YAML = Path("configs/step/stats.yaml")
_stats = StatsStep(**yaml.safe_load(_STATS_YAML.read_text()))

_TRAIN_YAML = Path("configs/step/train.yaml")
_train = TrainStep(**yaml.safe_load(_TRAIN_YAML.read_text()))

_FEATURES_YAML = Path("configs/step/features.yaml")
_features = FeaturesStep(**yaml.safe_load(_FEATURES_YAML.read_text()))

DATA_PATH = _stats.data_path
LOOKUP_PATH = _stats.lookup_path
BOROUGHS = _stats.group_values
MIN_ROWS_FOR_SAMPLING = _stats.min_rows_for_sampling
SAMPLE_FRACTION = _stats.per_group_sample_fraction
TIME_SLICE_START = _stats.time_slice_start
TIME_SLICE_END = _stats.time_slice_end
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

SAMPLE_SIZE = 200_000
RANDOM_STATE = _train.random_state
N_JOBS = _train.n_jobs
CV_FOLDS = _train.cv_folds
MODEL_FILE = _train.model_file
N_ESTIMATORS = _train.n_estimators
LEARNING_RATE = _train.learning_rate
NUM_LEAVES = _train.num_leaves
SUBSAMPLE = _train.subsample
COLSAMPLE_BYTREE = _train.colsample_bytree
QUANTILE_TAIL = _train.quantile_tail

ZONE_ID_COL = "LocationID"
ZONE_BOROUGH_COL = "Borough"
PICKUP_LOCATION_COL = "pickup_location_id"
DROPOFF_LOCATION_COL = "dropoff_location_id"
DATETIME_COL = "pickup_datetime"
TRIP_DISTANCE_COL = "trip_distance"
PASSENGER_COUNT_COL = "passenger_count"
TARGET_COL = "trip_duration_minutes"

PICKUP_BOROUGH_COL = "pickup_borough"
DURATION_COL = "trip_duration_minutes"

RESULTS_DIR = Path("results")
SAMPLE_CACHE = RESULTS_DIR / "joined_sample.parquet"
SAMPLE_META = RESULTS_DIR / "sample_meta.json"

_SAMPLE_PARAMS = ["SAMPLE_SIZE", "RANDOM_STATE", "PICKUP_BOROUGH_COL"]


def _params_hash() -> str:
    payload = {k: globals()[k] for k in _SAMPLE_PARAMS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def load_stratified_sample() -> "pd.DataFrame":
    import pandas as pd

    current_hash = _params_hash()
    if not SAMPLE_CACHE.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CACHE} not found. Run 'python learning/stats/00_prepare_data.py' first."
        )

    if SAMPLE_META.exists():
        meta = json.loads(SAMPLE_META.read_text())
        if meta.get("params_hash") != current_hash:
            logger.warning(
                "sample params changed (current=%s, cached=%s). "
                "Run 00_prepare_data.py to regenerate.",
                current_hash,
                meta.get("params_hash"),
            )

    return pd.read_parquet(SAMPLE_CACHE)


def load_boroughs_pandas():
    import pandas as pd
    df = pd.read_parquet(DATA_PATH)
    zones = pd.read_csv(LOOKUP_PATH)
    df = df.merge(
        zones[[ZONE_ID_COL, ZONE_BOROUGH_COL]],
        left_on=PICKUP_LOCATION_COL,
        right_on=ZONE_ID_COL,
        how="left",
    )
    return df.rename(columns={ZONE_BOROUGH_COL: PICKUP_BOROUGH_COL})


def get_spark_session(app_name="stats-learning", master="local[*]"):
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .getOrCreate()
    )


def _load_boroughs(spark):
    trips = spark.read.parquet(DATA_PATH)
    zones = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(LOOKUP_PATH)
    )
    return trips.join(
        zones.select(
            F.col(ZONE_ID_COL).alias(PICKUP_LOCATION_COL),
            F.col(_stats.group_column).alias(PICKUP_BOROUGH_COL),
        ),
        on=PICKUP_LOCATION_COL,
        how="left",
    )

