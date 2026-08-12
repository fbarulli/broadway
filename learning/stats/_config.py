import yaml
from pathlib import Path

from broadway.config.schema import FeaturesStep, StatsStep, TrainStep

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
