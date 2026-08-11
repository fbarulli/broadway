"""Load feature data → train model → save to disk."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.training.trainer import train

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.train:
        raise ValueError("training step requires dataset, experiment, and train config")
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    train_df = pd.read_parquet(out_dir / "train_features.parquet")
    target = cfg.dataset.target
    y = train_df[target]
    X = train_df.drop(columns=[col for col in train_df.columns if col == target or col not in train_df.select_dtypes(include="number").columns])
    model, elapsed = train(cfg.experiment.model.type, X, y, **cfg.experiment.model.params)
    logger.info(f"model trained in {elapsed:.1f}s")
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"model saved to {model_path}")
