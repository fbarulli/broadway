"""Orchestrates data layer — download → load → clean → split → save parquet."""

from __future__ import annotations

import logging
from pathlib import Path

from broadway.config.schema import PipelineConfig
from broadway.data.cleaner import clean
from broadway.data.loader import load
from broadway.data.splitter import split
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, write_record

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("etl step requires a dataset config")
    if not cfg.etl:
        raise ValueError("etl step requires an etl config")
    dataset = cfg.dataset
    df = load(dataset)
    rows_in = len(df)
    columns_before = list(df.columns)
    explained: list[tuple[str, int]] = []
    rs = cfg.experiment.random_state if cfg.experiment else cfg.etl.random_state
    if cfg.etl.ci_sample_size > 0:
        n_before = len(df)
        df = df.sample(n=min(cfg.etl.ci_sample_size, len(df)), random_state=rs)
        if len(df) < n_before:
            explained.append(("CI sampling", n_before - len(df)))
    df, clean_drops = clean(df, dataset)
    explained.extend(clean_drops)
    split_cfg = cfg.experiment.split if cfg.experiment else None
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    if split_cfg:
        train, val = split(df, dataset, split_cfg, random_state=rs)
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out_dir / cfg.etl.train_file, index=False)
        val.to_parquet(out_dir / cfg.etl.val_file, index=False)
        logger.info(f"saved train ({len(train)} rows) and val ({len(val)} rows)")
        rows_out = len(train) + len(val)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / cfg.etl.training_data_file, index=False)
        logger.info(f"saved training_data ({len(df)} rows)")
        rows_out = len(df)
    columns_after = list(df.columns)
    dropped_total = rows_in - rows_out
    explained_total = sum(n for _, n in explained)
    unexplained = max(0, dropped_total - explained_total)
    audit = TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=unexplained,
        reasons=[f"{reason}: -{n} rows" for reason, n in explained],
        columns_before=columns_before,
        columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )
    enforce_drop_fraction(audit, cfg.etl.max_drop_fraction)
    artifact = str(out_dir / (cfg.etl.train_file if split_cfg else cfg.etl.training_data_file))
    write_record(node_id("etl", dataset.name), "etl", artifact, [node_id("dataset", dataset.name)], audit=audit)
