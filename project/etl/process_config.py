"""Config for project/etl/process.py — reads taxi knobs from configs/project/taxi.yaml
and generic etl knobs from configs/step/etl.yaml."""

import yaml

from broadway.config.loader import CONFIGS_DIR
from broadway.config.schema import EtlStep, ProjectConfig

_PROJECT_YAML = CONFIGS_DIR / "project" / "taxi.yaml"
_project = ProjectConfig(**yaml.safe_load(_PROJECT_YAML.read_text()))

_ETL_YAML = CONFIGS_DIR / "step" / "etl.yaml"
_etl = EtlStep(**yaml.safe_load(_ETL_YAML.read_text()))

raw_dir: str = _project.raw_dir
processed_dir: str = _project.processed_dir
processed_file: str = _project.processed_file
ci_sample_size: int = _etl.ci_sample_size
min_trip_distance: float = _project.min_trip_distance
max_trip_distance: float = _project.max_trip_distance
min_trip_duration_minutes: float = _project.min_trip_duration_minutes
max_trip_duration_minutes: float = _project.max_trip_duration_minutes
min_pickup_datetime: str = _project.min_pickup_datetime
min_passenger_count: int = _project.min_passenger_count
max_passenger_count: int = _project.max_passenger_count
rename_map: dict[str, str] = _project.rename_map
random_state: int = _etl.random_state
