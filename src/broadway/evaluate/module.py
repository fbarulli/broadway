"""Load model → evaluate on holdout → log metrics → check promotion."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig
from broadway.evaluate.metrics import compute_metrics
from broadway.evaluate.promotion import should_promote
from broadway.utils import feature_columns

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.evaluate or not cfg.etl or not cfg.train:
        raise ValueError("evaluate step requires dataset, experiment, evaluate, etl, and train config")
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    val_path = out_dir / cfg.etl.val_features_file
    if not val_path.exists():
        val_path = out_dir / cfg.etl.train_features_file
        logger.warning(f"no validation set — evaluating on train set")
    val_df = pd.read_parquet(val_path)
    target = cfg.dataset.target
    y_true = val_df[target].values
    X_val = feature_columns(val_df, target)
    model_path = out_dir / cfg.train.model_file
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_true, y_pred)
    promote, reason = should_promote(metrics[cfg.evaluate.target_metric], None, cfg.evaluate.promotion_threshold)
    result = {"metrics": metrics, "promote": promote, "reason": reason}
    eval_dir = Path(cfg.evaluate.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / cfg.evaluate.output_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"evaluate: RMSE={metrics['rmse']:.4f}, promote={promote}")
