from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from broadway.analysis.contracts import AnalysisContract
from broadway.config.resolver import resolve_values
from broadway.config.schema import (
    BaselineStep,
    CausalStep,
    ContractsStep,
    DatasetContract,
    DiscoverStep,
    EdaStep,
    EnvironmentConfig,
    EtlStep,
    EvaluateStep,
    ExperimentConfig,
    FeaturesStep,
    FullStep,
    PipelineConfig,
    StatsStep,
    TrainStep,
)

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(os.getenv("BROADWAY_CONFIGS_DIR") or "configs")
DEFAULT_ENVIRONMENT = "development"

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
    "baseline": BaselineStep,
    "full": FullStep,
}

STEP_MODULES = {
    "discover": "broadway.discover.module",
    "etl": "broadway.etl.module",
    "contracts": "broadway.contracts.module",
    "eda": "broadway.eda.module",
    "features": "broadway.features.module",
    "stats": "broadway.stats.module",
    "causal": "broadway.causal.module",
    "train": "broadway.training.module",
    "evaluate": "broadway.evaluate.module",
    "baseline": "broadway.baseline.module",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(relative_path: str) -> Any:
    path = CONFIGS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"config file must be a mapping: {path}")
    return data


def _merge_section(merged: dict, section: str, name: str | None, optional: bool) -> None:
    if optional and name is None:
        return
    if not optional and name is None:
        raise ValueError(f"section '{section}' is required but name is None")
    raw = _load_yaml(f"{section}/{name}.yaml")
    merged.update({section: _deep_merge(merged.get(section, {}), raw)})


def _build_config(merged: dict, step: str) -> PipelineConfig:
    step_model = STEP_MODELS[step]
    merged = resolve_values(merged)
    config = PipelineConfig(
        analysis=AnalysisContract(**merged["analysis"]) if "analysis" in merged else None,
        dataset=DatasetContract(**merged["dataset"]) if "dataset" in merged else None,
        environment=EnvironmentConfig(**merged["environment"]),
        experiment=ExperimentConfig(**merged["experiment"]) if "experiment" in merged else None,
        **{step: step_model(**merged["step"])},
    )
    if step == "full" and config.full:
        for sub_step in config.full.steps:
            if sub_step not in STEP_MODELS or sub_step == "full" or sub_step == "discover":
                continue
            raw = _load_yaml(f"step/{sub_step}.yaml")
            resolved = resolve_values(raw)
            setattr(config, sub_step, STEP_MODELS[sub_step](**resolved))
    return config


def load_config(
    step: str,
    dataset: str | None = None,
    experiment: str | None = None,
    analysis: str | None = None,
    environment: str = DEFAULT_ENVIRONMENT,
) -> PipelineConfig:
    if step not in STEP_MODELS:
        raise ValueError(f"unknown step '{step}'. valid: {list(STEP_MODELS)}")
    logger.info(
        f"loading config — step={step}, dataset={dataset}, "
        f"experiment={experiment}, analysis={analysis}, environment={environment}"
    )
    merged: dict = {}
    _merge_section(merged, "environment", environment, optional=False)
    _merge_section(merged, "dataset", dataset, optional=True)
    _merge_section(merged, "experiment", experiment, optional=True)
    _merge_section(merged, "analysis", analysis, optional=True)
    _merge_section(merged, "step", step, optional=False)
    return _build_config(merged, step)
