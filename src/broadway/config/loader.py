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
    EnvironmentConfig,
    EtlStep,
    EvaluateStep,
    ExperimentConfig,
    FeaturesStep,
    FlowConfig,
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


def config_path(relative_path: str) -> Path:
    """Resolve a config file, preferring an optional project overlay."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"config path must be relative and cannot traverse parents: {relative_path}")
    overlay = os.getenv("BROADWAY_CONFIG_OVERLAY_DIR")
    if overlay:
        overlay_dir = Path(overlay)
        if not overlay_dir.is_dir():
            raise FileNotFoundError(f"config overlay directory not found: {overlay_dir}")
        overlay_path = overlay_dir / relative
        if overlay_path.is_file():
            return overlay_path
    return CONFIGS_DIR / relative


def _load_yaml(relative_path: str) -> Any:
    path = config_path(relative_path)
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
        for sub_step in resolve_full_steps(config):
            if sub_step not in STEP_MODELS or sub_step == "full" or sub_step == "discover":
                continue
            raw = _load_yaml(f"step/{sub_step}.yaml")
            resolved = resolve_values(raw)
            setattr(config, sub_step, STEP_MODELS[sub_step](**resolved))
    return config


def resolve_full_steps(cfg: PipelineConfig) -> list[str]:
    if cfg.analysis is None:
        raise ValueError("full step requires an analysis contract (--analysis)")
    if cfg.full is None:
        raise ValueError("full config missing")
    mode = cfg.analysis.mode.value
    if mode not in cfg.full.flows:
        raise ValueError(
            f"no flow defined for analysis mode '{mode}'. valid modes: {sorted(cfg.full.flows)}"
        )
    flow_name = cfg.full.flows[mode]
    try:
        raw = _load_yaml(f"flow/{flow_name}.yaml")
    except FileNotFoundError as exc:
        raise ValueError(f"flow '{flow_name}' not found for mode '{mode}': {exc}") from exc
    flow = FlowConfig(**raw)
    unknown = [s for s in flow.steps if s not in STEP_MODELS]
    if unknown:
        raise ValueError(
            f"flow '{flow_name}' lists unknown step(s) {unknown}. valid steps: {sorted(STEP_MODELS)}"
        )
    return flow.steps


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
