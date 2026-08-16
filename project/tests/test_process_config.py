"""Taxi-layer consistency: project/etl/process_config constants vs configs/project/taxi.yaml.

This is a taxi-demo test (not part of the platform suite): it verifies the
taxi ETL config module matches the taxi project YAML. Platform tests never
touch project-level data or configs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from broadway.config.schema import ProjectConfig
from project.etl.process_config import (
    max_trip_distance,
    max_trip_duration_minutes,
    min_trip_distance,
    min_trip_duration_minutes,
    rename_map,
)


def test_project_config_matches_process_constants() -> None:
    project = ProjectConfig(**yaml.safe_load(Path("configs/project/taxi.yaml").read_text()))
    assert project.min_trip_distance == min_trip_distance
    assert project.max_trip_distance == max_trip_distance
    assert project.min_trip_duration_minutes == min_trip_duration_minutes
    assert project.max_trip_duration_minutes == max_trip_duration_minutes
    assert project.rename_map == rename_map
