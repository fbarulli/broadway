"""Validate DataFrame against DatasetContract — columns and nulls (raw boundary).

The raw/input boundary checks column presence and null rates only. Dtype
conformance is enforced later at the canonical boundary by the ``etl`` step
(after structural cleaning normalizes representation, e.g. dates-as-strings).
"""

from __future__ import annotations

import logging

from broadway.config.schema import PipelineConfig
from broadway.contracts.checks import check_columns, check_nulls
from broadway.data.loader import load

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("contracts step requires a dataset config")
    if not cfg.contracts:
        raise ValueError("contracts step requires a contracts config")
    df = load(cfg.dataset)
    issues = check_columns(df, cfg.dataset) + check_nulls(
        df, cfg.dataset, cfg.contracts.null_threshold
    )
    if issues:
        logger.error("contracts check failed — %d issue(s)", len(issues))
        for issue in issues:
            logger.error("  %s", issue)
        raise ValueError(f"data contract violation: {len(issues)} issue(s)")
    logger.info("contracts check passed — all %d columns valid", len(df.columns))
