from __future__ import annotations

from dataclasses import dataclass

from broadway.config.schema import TrainingPipelineConfig

from ._config import _config as _cfg


@dataclass(frozen=True)
class TrainingConfig:
    train_test_cutoff: str
    random_state: int
    validation_size: float
    metric: str
    maximize: bool
    default_model: str
    supported_models: list[str]


def _build(cfg: TrainingPipelineConfig) -> TrainingConfig:
    return TrainingConfig(
        train_test_cutoff=cfg.train_test_cutoff,
        random_state=cfg.random_state,
        validation_size=cfg.validation_size,
        metric=cfg.metric,
        maximize=cfg.maximize,
        default_model=cfg.default_model,
        supported_models=cfg.supported_models,
    )


training: TrainingConfig = _build(_cfg.training_pipeline)
