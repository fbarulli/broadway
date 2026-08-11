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

logger = logging.getLogger(__name__)


def run(cfg: PipelineConfig) -> None:
    if not cfg.dataset or not cfg.experiment or not cfg.evaluate:
        raise ValueError("evaluate step requires dataset, experiment, and evaluate config")
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    val_path = out_dir / "val_features.parquet"
    if not val_path.exists():
        logger.warning("no validation set found — using train set for evaluation")
        val_path = out_dir / "train_features.parquet"
    val_df = pd.read_parquet(val_path)
    target = cfg.dataset.target
    y_true = val_df[target].values
    X_val = val_df.drop(columns=[col for col in val_df.columns if col == target or col not in val_df.select_dtypes(include="number").columns])
    model_path = out_dir / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_true, y_pred)
    promote, reason = should_promote(metrics["rmse"], None, cfg.evaluate.promotion_threshold)
    result = {"metrics": metrics, "promote": promote, "reason": reason}
    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"evaluate: RMSE={metrics['rmse']:.4f}, promote={promote}")
