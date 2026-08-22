"""Compose the experiment's Pipeline and fit it."""

from __future__ import annotations

import time
from collections.abc import Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from broadway.config.schema import PipelineConfig
from broadway.features.recipe import build_pipeline
from broadway.training.contracts import TrainingResult
from broadway.training.models.registry import get_model

_PRE_PARAM_PREFIX = "pre__"


def build_model_pipeline(
    cfg: PipelineConfig,
    model_type: str,
    params: Mapping[str, float | int | str],
) -> Pipeline:
    """Compose preprocessing (or passthrough) with the registry model.

    Model params are passed bare; preprocessing params use the ``pre__``
    prefix (``pre__<step>__<param>``) and are applied after composition.
    Shared by the trainer and the HPO objective — one composition, no drift.
    """
    if cfg.experiment is None:
        raise ValueError("model pipeline requires an experiment config")
    model_params = {
        key: value for key, value in params.items() if not key.startswith(_PRE_PARAM_PREFIX)
    }
    pipeline = Pipeline(
        [
            ("pre", build_pipeline(cfg)),
            ("model", get_model(model_type, **model_params)),
        ]
    )
    pre_params = {
        key: value for key, value in params.items() if key.startswith(_PRE_PARAM_PREFIX)
    }
    if pre_params:
        pipeline.set_params(**pre_params)
    return pipeline


def train(
    cfg: PipelineConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **params: float | str,
) -> tuple[Pipeline, TrainingResult]:
    """Build the experiment's Pipeline from config, fit it, and record the result."""
    if cfg.experiment is None:
        raise ValueError("train requires an experiment config")
    model = build_model_pipeline(cfg, cfg.experiment.model.type, params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, TrainingResult(
        model_type=cfg.experiment.model.type,
        params=params,
        train_time_seconds=round(elapsed, 3),
    )
