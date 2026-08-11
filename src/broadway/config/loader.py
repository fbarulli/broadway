from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import yaml

from broad_way.config.schema import (
    CausalStep,
    ContractsStep,
    DatasetContract,
    DiscoverStep,
    EdaStep,
    EnvironmentConfig,
    EvaluateStep,
    ExperimentConfig,
    FeaturesStep,
    FullStep,
    PipelineConfig,
    StatsStep,
    TrainStep,
)

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path("configs")

STEP_MODELS = {
    "discover": DiscoverStep,
    "etl": EtlStep,
    "contracts": ContractsStep,
    "eda": EdaStep,
    "features": FeaturesStep,
    "stats": StatsStep,
    "causal": CausalStep,
    "train": TrainStep,
    "evaluate": EvaluateStep,
    "full": FullStep,
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(relative_path: str) -> dict:
    path = CONFIGS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    step: str,
    dataset: str | None = None,
    experiment: str | None = None,
    environment: str = "development",
) -> PipelineConfig:
    if step not in STEP_MODELS:
        raise ValueError(f"unknown step '{step}'. valid: {list(STEP_MODELS)}")

    logger.info(
        f"loading config — step={step}, dataset={dataset}, "
        f"experiment={experiment}, environment={environment}"
    )

    merged: dict = {}

    env_raw = _load_yaml(f"environment/{environment}.yaml")
    merged = _deep_merge(merged, {"environment": env_raw})

    if dataset is not None:
        dataset_raw = _load_yaml(f"dataset/{dataset}.yaml")
        merged = _deep_merge(merged, {"dataset": dataset_raw})

    if experiment is not None:
        experiment_raw = _load_yaml(f"experiment/{experiment}.yaml")
        merged = _deep_merge(merged, {"experiment": experiment_raw})

    step_raw = _load_yaml(f"step/{step}.yaml")
    merged = _deep_merge(merged, {"step": step_raw})

    step_model = STEP_MODELS[step]

    return PipelineConfig(
        dataset=DatasetContract(**merged.get("dataset")) if merged.get("dataset") else None,
        environment=EnvironmentConfig(**merged["environment"]),
        experiment=ExperimentConfig(**merged.get("experiment")) if merged.get("experiment") else None,
        **{step: step_model(**merged["step"])},
    )
