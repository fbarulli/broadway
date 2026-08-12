"""Config for etl/process.py — reads from configs/step/etl.yaml."""

from pathlib import Path

import yaml

from broadway.config.schema import EtlStep

_ETL_YAML = Path("configs/step/etl.yaml")
_cfg = EtlStep(**yaml.safe_load(_ETL_YAML.read_text()))

raw_dir: str = _cfg.raw_dir
processed_dir: str = _cfg.processed_dir
processed_file: str = _cfg.processed_file
ci_sample_size: int = _cfg.ci_sample_size
min_trip_distance: float = _cfg.min_trip_distance
max_trip_distance: float = _cfg.max_trip_distance
min_trip_duration_minutes: float = _cfg.min_trip_duration_minutes
max_trip_duration_minutes: float = _cfg.max_trip_duration_minutes
rename_map: dict[str, str] = _cfg.rename_map
random_state: int = _cfg.random_state
