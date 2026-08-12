from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from broadway.config.schema import DataPipelineConfig

from ._config import _config as _cfg


@dataclass(frozen=True)
class DataConfig:
    raw_data_dir: Path
    processed_dir: Path
    processed_file: str
    taxi_lookup: Path
    taxi_urls: tuple[str, ...]
    lookup_url: str
    training_table: str
    target: str
    validation_cutoff: str
    min_trip_distance: float
    max_trip_distance: float
    min_trip_duration_minutes: float
    max_trip_duration_minutes: float
    ci_sample_size: int
    batch_size: int
    rename_map: dict[str, str]
    encoding_smoothing: int
    frequency_fill: float


def _build(cfg: DataPipelineConfig) -> DataConfig:
    raw_data_dir = Path(cfg.raw_dir)
    return DataConfig(
        raw_data_dir=raw_data_dir,
        processed_dir=Path(cfg.processed_dir),
        processed_file=cfg.processed_file,
        taxi_lookup=raw_data_dir / cfg.lookup_filename,
        taxi_urls=tuple(cfg.taxi_urls),
        lookup_url=cfg.lookup_url,
        training_table=cfg.training_table,
        target=cfg.target,
        validation_cutoff=cfg.validation_cutoff,
        min_trip_distance=cfg.min_trip_distance,
        max_trip_distance=cfg.max_trip_distance,
        min_trip_duration_minutes=cfg.min_trip_duration_minutes,
        max_trip_duration_minutes=cfg.max_trip_duration_minutes,
        ci_sample_size=cfg.ci_sample_size,
        batch_size=cfg.batch_size,
        rename_map=cfg.rename_map,
        encoding_smoothing=cfg.encoding_smoothing,
        frequency_fill=cfg.frequency_fill,
    )


data: DataConfig = _build(_cfg.data_pipeline)
