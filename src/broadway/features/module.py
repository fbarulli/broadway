"""Fit FeaturePipeline on train → transform train + test → save pipeline."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.features.generic import build_generic_feature_specs
from broadway.features.pipeline import FeaturePipeline
from broadway.features.plan import ColumnPlan
from broadway.features.schema import build_engineered_schema
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, write_record

logger = logging.getLogger(__name__)


def _load_split(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    assert cfg.etl is not None
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    train = pd.read_parquet(out_dir / cfg.etl.train_file)
    val_path = out_dir / cfg.etl.val_file
    val = pd.read_parquet(val_path) if val_path.exists() else None
    return train, val


def _plan_ordered(frame: pd.DataFrame, plan: ColumnPlan) -> pd.DataFrame:
    """Put the plan's columns first, in plan order; keep extra columns after.

    The read contract (``engineered_schema_for``) is ordered: the persisted
    feature file must carry the model surface in plan order or a real
    train/evaluate read fails loud. ``FeaturePipeline.transform`` appends
    derived/encoded columns to the joined frame, so the write reorders here —
    extras (the target, joined columns) are tolerated and follow.
    """
    declared = list(plan.names)
    extras = [name for name in frame.columns if name not in declared]
    return frame.loc[:, declared + extras]


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.features or not cfg.etl:
        raise ValueError("features step requires dataset, experiment, features, and etl config")
    train, val = _load_split(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    pipeline = FeaturePipeline()
    pipeline.fit(train, plan, cfg.dataset.target)
    rows_in = len(train)
    columns_before = list(train.columns)
    train_out = _plan_ordered(
        pipeline.transform(train, plan, cfg.dataset.target, cfg.features.frequency_fill), plan
    )
    specs = build_generic_feature_specs(cfg.dataset, plan)
    build_engineered_schema(specs).validate(train_out)
    rows_out = len(train_out)
    columns_after = list(train_out.columns)
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    train_out.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    logger.info(f"train features written ({len(train_out)} rows)")
    if val is not None:
        val_out = _plan_ordered(
            pipeline.transform(val, plan, cfg.dataset.target, cfg.features.frequency_fill), plan
        )
        build_engineered_schema(specs).validate(val_out)
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
