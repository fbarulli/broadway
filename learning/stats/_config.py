import yaml
from pathlib import Path

_STATS_YAML = Path("configs/step/stats.yaml")
_stats = yaml.safe_load(_STATS_YAML.read_text())

DATA_PATH = _stats.get("data_path", "data/processed/training_data.parquet")
LOOKUP_PATH = _stats.get("lookup_path", "data/raw/taxi_zone_lookup.csv")
BOROUGHS = _stats["group_values"]
MIN_ROWS_FOR_SAMPLING = _stats["min_rows_for_sampling"]
SAMPLE_FRACTION = _stats["per_group_sample_fraction"]
TIME_SLICE_START = _stats.get("time_slice_start", "2024-01-01")
TIME_SLICE_END = _stats.get("time_slice_end", "2024-01-31")
TIME_SPLIT_CUTOFF = _stats.get("time_split_cutoff", "2024-02-15")
ACF_LAGS = _stats.get("acf_lags", 48)
