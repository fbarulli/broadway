from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from broadway.config.loader import load_config
from broadway.config.schema import (
    ExperimentConfig,
    FeatureConfig,
    HPOConfig,
    LookupSpec,
    ModelConfig,
    ProjectConfig,
    SplitConfig,
)
from broadway.etl.process_config import (
    min_trip_duration_minutes,
    max_trip_duration_minutes,
    min_trip_distance,
    max_trip_distance,
    rename_map,
)


@pytest.fixture
def train_cfg():
    cfg = load_config("train", dataset="taxi", experiment="taxi")
    assert cfg.train is not None
    return cfg.train


@pytest.fixture
def evaluate_cfg():
    cfg = load_config("evaluate", dataset="taxi", experiment="taxi")
    assert cfg.evaluate is not None
    return cfg.evaluate


@pytest.fixture
def etl_cfg():
    cfg = load_config("etl", dataset="taxi", experiment="taxi")
    assert cfg.etl is not None
    return cfg.etl


@pytest.fixture
def stats_cfg():
    cfg = load_config("stats", dataset="taxi", experiment="taxi")
    assert cfg.stats is not None
    return cfg.stats


def test_load_train_config(train_cfg) -> None:
    assert train_cfg.model_file
    assert train_cfg.random_state


def test_load_evaluate_config(evaluate_cfg) -> None:
    assert evaluate_cfg.target_metric
    assert evaluate_cfg.promotion_threshold


def test_load_etl_with_pipeline_fields(etl_cfg) -> None:
    assert etl_cfg.train_file
    assert etl_cfg.max_drop_fraction >= 0.0


def test_project_config_matches_process_constants() -> None:
    project = ProjectConfig(**yaml.safe_load(Path("configs/project/taxi.yaml").read_text()))
    assert project.min_trip_distance == min_trip_distance
    assert project.max_trip_distance == max_trip_distance
    assert project.min_trip_duration_minutes == min_trip_duration_minutes
    assert project.max_trip_duration_minutes == max_trip_duration_minutes
    assert project.rename_map == rename_map


def test_missing_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("train", dataset="nonexistent", experiment="taxi")


def test_missing_experiment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("full", dataset="taxi", experiment="nonexistent")


def test_invalid_step_raises() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        load_config("bogus")


def test_stats_config_fields(stats_cfg) -> None:
    assert stats_cfg.min_rows_for_sampling
    assert stats_cfg.per_group_sample_fraction


def _experiment_config(model_type: str, search_space: dict[str, list[float | int]]) -> ExperimentConfig:
    return ExperimentConfig(
        features=FeatureConfig(include=["rooms"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type=model_type, params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=HPOConfig(engine="optuna", trials=10, search_space=search_space),
    )


def test_hpo_search_space_valid_for_lgbm() -> None:
    config = _experiment_config(
        "lgbm",
        {"max_depth": [3, 10], "learning_rate": [0.01, 0.3]},
    )
    assert config.hpo is not None
    assert set(config.hpo.search_space) == {"max_depth", "learning_rate"}


def test_hpo_search_space_invalid_for_linear_raises() -> None:
    with pytest.raises(ValidationError):
        _experiment_config("linear", {"max_depth": [3, 10]})


def test_lookup_spec_value_policies_parse() -> None:
    spec = LookupSpec(**yaml.safe_load(
        """
        path: lookup.csv
        key: LocationID
        value_policies:
          Borough:
            sentinel_values:
              - Unknown
        """
    ))
    assert spec.value_policies["Borough"].sentinel_values == ["Unknown"]


def test_lookup_spec_value_policies_default_empty() -> None:
    spec = LookupSpec(path="lookup.csv", key="LocationID")
    assert spec.value_policies == {}
