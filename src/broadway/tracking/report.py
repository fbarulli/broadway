"""``ds-pipeline tracking`` — pure-render the published tracking surface.

Reads the ignored machine truth (``baseline_stats.json`` plus the optional
profile/clean evidence) and writes the tracked ``reports/tracking/data_card.md``.
Never re-runs etl/features/train and never reads parquet: the same
report/audit renderer pattern as ``ds-pipeline audit``.
"""

from __future__ import annotations

from broadway.config.loader import load_config
from broadway.data.loader import ArtifactStore, ProcessedStore
from broadway.tracking.bundle import TrackingBundle
from broadway.tracking.manifest import tracking_dir


def run(dataset: str, environment: str) -> None:
    """Render ``data_card.md`` from persisted tracking JSON; prints the path."""
    cfg = load_config("train", dataset=dataset, environment=environment)
    if cfg.dataset is None or cfg.train is None or cfg.train.tracking is None:
        raise ValueError("tracking report requires a train config with a tracking section")
    processed_dir = ProcessedStore.from_config(cfg).directory
    bundle = TrackingBundle.from_config(
        dataset=cfg.dataset.name,
        config=cfg.train.tracking,
        clean_result_path=processed_dir / f"{cfg.dataset.name}_clean.json",
        machine_dir=ArtifactStore.from_contract(cfg.dataset, tracking_dir()).directory,
    )
    path = bundle.publish_data_card()
    print(f"wrote {path}")
