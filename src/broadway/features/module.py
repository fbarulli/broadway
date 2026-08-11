"""Fit FeaturePipeline on train → transform train + test → save pipeline."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


def _load_split(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    train = pd.read_parquet(out_dir / cfg.etl.train_file)
    val_path = out_dir / cfg.etl.val_file
    val = pd.read_parquet(val_path) if val_path.exists() else None
    return train, val


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.features or not cfg.etl:
        raise ValueError("features step requires dataset, experiment, features, and etl config")
    train, val = _load_split(cfg)
    pipeline = FeaturePipeline(encodings=cfg.experiment.features.encodings)
    pipeline.fit(train, cfg.dataset.target, cfg.features.encoding_smoothing)
    train_out = pipeline.transform(train, cfg.experiment.features, cfg.dataset.target, cfg.features.frequency_fill)
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    train_out.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    logger.info(f"train features written ({len(train_out)} rows)")
    if val is not None:
        val_out = pipeline.transform(val, cfg.experiment.features, cfg.dataset.target, cfg.features.frequency_fill)
        val_out.to_parquet(out_dir / cfg.etl.val_features_file, index=False)
        logger.info(f"val features written ({len(val_out)} rows)")
    pipeline_path = out_dir / cfg.features.pipeline_file
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"pipeline saved to {pipeline_path}")
