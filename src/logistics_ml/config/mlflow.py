from __future__ import annotations

from dataclasses import dataclass

from broadway.config.schema import MLflowServiceConfig

from ._config import _config as _cfg


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    champion_alias: str


def _build(cfg: MLflowServiceConfig) -> MLflowConfig:
    return MLflowConfig(
        tracking_uri=cfg.tracking_uri,
        experiment_name=cfg.experiment_name,
        registered_model_name=cfg.registered_model_name,
        champion_alias=cfg.champion_alias,
    )


mlflow: MLflowConfig = _build(_cfg.mlflow_service)
