"""HPO objective for the project dataset — the real data, no fixtures.

Adapter contract (``gpu_training/hpo_kernel.py --objective``)::

    build(study: Mapping[str, Any]) -> objective(params, trial) -> float

Every dial comes from the project config, never from this file:

* the project dataset contract the study names — the parquet path, the
  target, the datetime column and the feature columns (by role);
* the project experiment contract the study names — the split kind/size and seed;
* the project HPO spec (``hpo_spec`` in the study config) — the model, its
  search space and the target metric.

The objective is the canonical ``broadway.training.hpo.make_objective``: it
composes the experiment Pipeline through ``build_model_pipeline`` and scores the
validation slice, so the lane never re-implements a fit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Importing the project package activates its config overlay
# (BROADWAY_CONFIG_OVERLAY_DIR), so load_config resolves project/config.
import project  # noqa: F401
from broadway.config.loader import load_config
from broadway.config.schema import ColumnRole, DatasetContract, ExperimentConfig, HPOConfig
from broadway.data.splitter import split
from broadway.training.hpo import Objective, make_objective

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dataset_path(contract: DatasetContract) -> Path:
    """The declared parquet, anchored at the repo root unless it is absolute."""
    path = Path(contract.path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _feature_columns(contract: DatasetContract) -> list[str]:
    return [
        name for name, column in contract.columns.items() if column.role is ColumnRole.FEATURE
    ]


def _slices(
    contract: DatasetContract, experiment: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Read the declared columns and split them through the canonical splitter."""
    features = _feature_columns(contract)
    needed = [*features, contract.target]
    if contract.datetime_column:
        needed.append(contract.datetime_column)
    frame = pd.read_parquet(_dataset_path(contract), columns=needed)
    train, validation = split(frame, contract, experiment.split, experiment.random_state)
    return (
        train[features],
        train[contract.target],
        validation[features],
        validation[contract.target],
    )


def _hpo_config(spec_path: str) -> HPOConfig:
    raw = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("hpo"), dict):
        raise ValueError(f"HPO spec {spec_path!r} has no 'hpo' block")
    return HPOConfig(**raw["hpo"])


def build(study: Mapping[str, Any]) -> Objective:
    """Build the project objective from the study's config references."""
    cfg = load_config(
        "train", dataset=str(study["dataset"]), experiment=str(study["experiment"])
    )
    if cfg.dataset is None or cfg.experiment is None:
        raise ValueError("study dataset/experiment configs did not load a dataset and experiment")
    hpo = _hpo_config(str(study["hpo_spec"]))
    model = str(study["model"])
    X_train, y_train, X_val, y_val = _slices(cfg.dataset, cfg.experiment)
    return make_objective(cfg, model, hpo.target_metric, X_train, y_train, X_val, y_val)
