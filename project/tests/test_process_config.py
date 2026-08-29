"""Taxi-layer consistency: project/etl/process_config constants vs project/config/project/taxi.yaml.

This is a taxi-demo test (not part of the platform suite): it verifies the
taxi ETL config module matches the taxi project YAML. Platform tests never
touch project-level data or configs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from broadway.config.loader import config_path
from project.config import ProjectConfig
from project.etl.process_config import (
    max_trip_distance,
    max_trip_duration_minutes,
    min_trip_distance,
    min_trip_duration_minutes,
    rename_map,
)


def test_project_config_matches_process_constants() -> None:
    path = config_path("project/taxi.yaml")
    assert path == Path("project/config/project/taxi.yaml").resolve()
    project = ProjectConfig(**yaml.safe_load(path.read_text()))
    assert project.min_trip_distance == min_trip_distance
    assert project.max_trip_distance == max_trip_distance
    assert project.min_trip_duration_minutes == min_trip_duration_minutes
    assert project.max_trip_duration_minutes == max_trip_duration_minutes
    assert project.rename_map == rename_map
