"""Fit FeaturePipeline on train → transform train + test → save pipeline."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.features.pipeline import FeaturePipeline
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, write_record

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
    rows_in = len(train)
    columns_before = list(train.columns)
    train_out = pipeline.transform(train, cfg.experiment.features, cfg.dataset.target, cfg.features.frequency_fill)
    rows_out = len(train_out)
    columns_after = list(train_out.columns)
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
    dropped_total = rows_in - rows_out
    audit = TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=max(0, dropped_total),
        reasons=[] if dropped_total == 0 else [f"unexpected row loss: {dropped_total} rows"],
        columns_before=columns_before,
        columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )
    enforce_drop_fraction(audit, cfg.features.max_drop_fraction)
    write_record(node_id("features", cfg.dataset.name), "features", str(pipeline_path), [node_id("etl", cfg.dataset.name)], audit=audit)
