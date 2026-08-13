"""Orchestrates data layer — load → sample → canonicalize → validate → save → split."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from broadway.cleaning.models import StructuralCleanResult
from broadway.config.schema import PipelineConfig
from broadway.contracts.pandera import build_raw_schema
from broadway.contracts.selectors import datetime_columns
from broadway.data.cleaner import canonicalize
from broadway.data.loader import load
from broadway.data.splitter import split
from broadway.lineage.ids import node_id
from broadway.lineage.models import TransformAudit
from broadway.lineage.records import enforce_drop_fraction, write_record

logger = logging.getLogger(__name__)


def _explained_rows(reasons: list[str]) -> int:
    total = 0
    for reason in reasons:
        match = re.search(r"-(\d+) rows$", reason)
        if match:
            total += int(match.group(1))
    return total


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("etl step requires a dataset config")
    if not cfg.etl:
        raise ValueError("etl step requires an etl config")
    dataset = cfg.dataset
    df = load(dataset)
    rows_in = len(df)
    columns_before = list(df.columns)

    reasons: list[str] = []
    rs = cfg.experiment.random_state if cfg.experiment else cfg.etl.random_state
    if cfg.etl.ci_sample_size > 0:
        n_before = len(df)
        df = df.sample(n=min(cfg.etl.ci_sample_size, len(df)), random_state=rs)
        if len(df) < n_before:
            reasons.append(f"CI sampling: -{n_before - len(df)} rows")

    df, clean_reasons, parse_failures, observed_missing = canonicalize(
        df,
        target=dataset.target,
        datetime_columns=datetime_columns(dataset),
        missing_encodings=cfg.etl.missing_encodings,
    )
    reasons.extend(clean_reasons)

    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    build_raw_schema(dataset).validate(df)

    canonical_path = out_dir / f"{dataset.name}_canonical.parquet"
    df.to_parquet(canonical_path, index=False)
    logger.info(f"saved canonical ({len(df)} rows) to {canonical_path}")

    split_cfg = cfg.experiment.split if cfg.experiment else None
    if split_cfg:
        train, val = split(df, dataset, split_cfg, random_state=rs)
        train.to_parquet(out_dir / cfg.etl.train_file, index=False)
        val.to_parquet(out_dir / cfg.etl.val_file, index=False)
        logger.info(f"saved train ({len(train)} rows) and val ({len(val)} rows)")
        rows_out = len(train) + len(val)
    else:
        df.to_parquet(out_dir / cfg.etl.training_data_file, index=False)
        logger.info(f"saved training_data ({len(df)} rows)")
        rows_out = len(df)

    columns_after = list(df.columns)
    dropped_total = rows_in - rows_out
    unexplained = max(0, dropped_total - _explained_rows(reasons))
    audit = TransformAudit(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped_total=dropped_total,
        rows_dropped_unexplained=unexplained,
        reasons=reasons,
        columns_before=columns_before,
        columns_after=columns_after,
        columns_added=sorted(set(columns_after) - set(columns_before)),
        columns_removed=sorted(set(columns_before) - set(columns_after)),
    )
    enforce_drop_fraction(audit, cfg.etl.max_drop_fraction)

    result = StructuralCleanResult(
        audit=audit,
        parse_failures=parse_failures,
        missing_encodings=observed_missing,
        canonical_path=str(canonical_path),
    )
    result_path = out_dir / f"{dataset.name}_clean.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"saved structural clean result to {result_path}")

    write_record(
        node_id("etl", dataset.name),
        "etl",
        str(canonical_path),
        [node_id("dataset", dataset.name)],
        audit=audit,
    )
