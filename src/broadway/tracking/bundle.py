"""The train-side Data Tracking bundle: baseline truth and data card.

Owns the production wiring for the machine-truth writer
(``baseline_stats.json`` from the training frame) and the pure-render published
surface (``data_card.md`` from persisted JSON). One class owns both so the
train step and the ``ds-pipeline tracking`` report command share the same
resolution of paths and knobs — no second copy of the bundle contract.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from broadway.cleaning.models import StructuralCleanResult
from broadway.config.schema import TrackingConfig
from broadway.discover.profile import DatasetProfile
from broadway.lineage.models import TransformAudit
from broadway.tracking.baseline import (
    BaselineOptions,
    DataCardOptions,
    build_baseline_stats,
    load_baseline_stats,
    render_data_card,
    write_baseline_stats,
    write_data_card,
)

logger = logging.getLogger(__name__)


class TrackingBundle:
    """Resolve, write, and render the tracking bundle for one dataset."""

    def __init__(
        self,
        *,
        dataset: str,
        config: TrackingConfig,
        clean_result_path: Path,
        machine_dir: Path,
        reports_dir: Path,
    ) -> None:
        self._dataset = dataset
        self._config = config
        self._clean_result_path = clean_result_path
        self._machine_dir = machine_dir
        self._reports_dir = reports_dir

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        config: TrackingConfig,
        clean_result_path: Path,
        machine_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> TrackingBundle:
        """Build from config; the tracking env override wins for test/lane sandboxes."""
        return cls(
            dataset=dataset,
            config=config,
            clean_result_path=clean_result_path,
            machine_dir=machine_dir or Path(os.getenv("BROADWAY_TRACKING_DIR", config.machine_dir)),
            reports_dir=reports_dir or Path(os.getenv("BROADWAY_REPORTS_DIR", config.reports_dir)),
        )

    @property
    def machine_dir(self) -> Path:
        return self._machine_dir

    @property
    def baseline_path(self) -> Path:
        return self._machine_dir / self._config.baseline.output_file

    @property
    def data_card_path(self) -> Path:
        return self._reports_dir / self._config.data_card.output_file

    def record_training_truth(self, train_df: pd.DataFrame) -> Path:
        """Write ``baseline_stats.json`` for the training frame via the canonical writer."""
        stats = build_baseline_stats(
            train_df,
            dataset=self._dataset,
            options=BaselineOptions(
                n_bins=self._config.baseline.n_bins,
                top_k=self._config.baseline.top_k,
                smoothing=self._config.baseline.smoothing,
            ),
        )
        path = write_baseline_stats(stats, self.baseline_path)
        logger.info(
            "tracking: wrote baseline_stats for %s (%d rows, %d columns) to %s",
            self._dataset,
            stats.row_count,
            len(stats.features),
            path,
        )
        return path

    def publish_data_card(self) -> Path:
        """Pure-render ``data_card.md`` from the persisted JSON bundle."""
        try:
            baseline = load_baseline_stats(self.baseline_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"tracking baseline not found at {self.baseline_path} — run the train step first"
            ) from exc
        markdown = render_data_card(
            dataset=self._dataset,
            profile=self._load_profile(),
            audits=self._load_audits(),
            baseline=baseline,
            options=DataCardOptions(
                biases=list(self._config.data_card.biases),
                missing_flag_rate=self._config.data_card.missing_flag_rate,
                generated_at=self._config.data_card.generated_at,
            ),
        )
        path = write_data_card(markdown, self.data_card_path)
        logger.info("tracking: wrote data card for %s to %s", self._dataset, path)
        return path

    def _load_profile(self) -> DatasetProfile | None:
        """Profile evidence for date range + missingness; a foreign profile is ignored."""
        path = Path(self._config.profile_file)
        if not path.exists():
            logger.info("tracking: no profile at %s — data card falls back to baseline", path)
            return None
        profile = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
        if profile.name != self._dataset:
            logger.warning(
                "tracking: profile at %s is for dataset '%s', not '%s' — using baseline only",
                path,
                profile.name,
                self._dataset,
            )
            return None
        return profile

    def _load_audits(self) -> list[TransformAudit]:
        if not self._clean_result_path.exists():
            logger.info(
                "tracking: no clean result at %s — data card exclusions are 'none recorded'",
                self._clean_result_path,
            )
            return []
        result = StructuralCleanResult.model_validate_json(
            self._clean_result_path.read_text(encoding="utf-8")
        )
        return [result.audit]
