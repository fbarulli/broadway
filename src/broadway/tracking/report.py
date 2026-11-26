"""``ds-pipeline tracking`` — pure-render the published tracking surface.

Reads the ignored machine truth (``baseline_stats.json`` plus the optional
profile/clean evidence) and writes the tracked ``reports/tracking/data_card.md``.
Never re-runs etl/features/train and never reads parquet: the same
report/audit renderer pattern as ``ds-pipeline audit``.
"""

from __future__ import annotations

from pathlib import Path

from broadway.config.loader import load_config
from broadway.tracking.bundle import TrackingBundle


def run(dataset: str, environment: str) -> None:
    """Render ``data_card.md`` from persisted tracking JSON; prints the path."""
    cfg = load_config("train", dataset=dataset, environment=environment)
    if cfg.dataset is None or cfg.train is None or cfg.train.tracking is None:
        raise ValueError("tracking report requires a train config with a tracking section")
    processed_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    bundle = TrackingBundle.from_config(
        dataset=cfg.dataset.name,
        config=cfg.train.tracking,
        clean_result_path=processed_dir / f"{cfg.dataset.name}_clean.json",
    )
    path = bundle.publish_data_card()
    print(f"wrote {path}")
