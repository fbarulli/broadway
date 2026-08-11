"""Orchestrates data layer — download → load → clean → split → save parquet."""

from __future__ import annotations

import logging
from pathlib import Path

from broadway.config.schema import PipelineConfig
from broadway.data.cleaner import clean
from broadway.data.loader import load
from broadway.data.splitter import split

logger = logging.getLogger(__name__)

TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
TRAINING_DATA_FILE = "training_data.parquet"


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("etl step requires a dataset config")
    if not cfg.etl:
        raise ValueError("etl step requires an etl config")
    dataset = cfg.dataset
    df = load(dataset)
    rs = cfg.experiment.random_state if cfg.experiment else cfg.etl.random_state
    if cfg.etl.ci_sample_size > 0:
        df = df.sample(n=min(cfg.etl.ci_sample_size, len(df)), random_state=rs)
    df = clean(df, dataset)
    split_cfg = cfg.experiment.split if cfg.experiment else None
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    if split_cfg:
        train, val = split(df, dataset, split_cfg, random_state=rs)
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out_dir / TRAIN_FILE, index=False)
        val.to_parquet(out_dir / VAL_FILE, index=False)
        logger.info(f"saved train ({len(train)} rows) and val ({len(val)} rows)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / TRAINING_DATA_FILE, index=False)
        logger.info(f"saved training_data ({len(df)} rows)")
