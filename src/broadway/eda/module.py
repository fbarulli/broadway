"""Orchestrates EDA submodules → produces HTML report."""

from __future__ import annotations

import logging
from pathlib import Path

from broadway.config.schema import PipelineConfig
from broadway.data.loader import load
from broadway.eda.compare import compare
from broadway.eda.missing import null_counts, null_patterns, littles_mcar_test
from broadway.eda.quality import class_imbalance, constant_columns, duplicate_rows, outlier_counts_iqr
from broadway.eda.report import build_report
from broadway.eda.summary import summarize
from broadway.eda.visualize import boxplot, correlation_heatmap, histogram

logger = logging.getLogger(__name__)


def _build_figures(df, cfg):
    figs = []
    for col in df.select_dtypes(include="number").columns[:4]:
        figs.append(histogram(df, col))
        figs.append(boxplot(df, col))
    corr = correlation_heatmap(df)
    if corr.data:
        figs.append(corr)
    return figs


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset:
        raise ValueError("eda step requires a dataset config")
    df = load(cfg.dataset)
    quality = {
        "constant_columns": constant_columns(df),
        "duplicate_rows": duplicate_rows(df),
        "outliers_iqr": outlier_counts_iqr(df),
        "class_imbalance": class_imbalance(df, cfg.dataset.target),
    }
    missing = {
        "null_counts": null_counts(df),
        "null_patterns": null_patterns(df).to_dict(orient="records"),
        "littles_mcar": littles_mcar_test(df),
    }
    figs = _build_figures(df, cfg)
    out_path = Path(cfg.eda.output_dir) / "eda.html" if cfg.eda else Path("artifacts/reports/eda.html")
    build_report(summarize(df), quality, missing, figs, out_path)
    logger.info(f"eda report written to {out_path}")
