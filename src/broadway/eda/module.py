"""EDA pipeline step — orchestrates summary, missingness, quality, decisions and report.

The profiled frame is ONE proportional draw of the train+val splits taken while
reading (a bounded batch at a time, never the full 8.5M-row frame): sampling the
splits independently would turn an 80/10 split into a 50/50 one and make the
before/after comparison a composition artifact. Every decision then re-measures
inside semantic segments (``eda.segments``) and records the n and seed actually
used.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.figure import Figure

from broadway.config.schema import DatasetContract, EdaStep, EtlStep, PipelineConfig
from broadway.data.loader import canonical_path
from broadway.eda.artifacts import EdaArtifactWriter
from broadway.eda.compare import EtlComparator
from broadway.eda.correlation import CorrelationAnalyzer
from broadway.eda.figures import DecisionFigures
from broadway.eda.missing import MissingnessAnalyzer
from broadway.eda.models import (
    CorrelationResult,
    EdaDecisionBundle,
    EdaResult,
    EmbeddedFigure,
    FeatureTestBundle,
    MissingResult,
    OutlierResult,
    SummaryResult,
)
from broadway.eda.outliers import OutlierDecisionAnalyzer
from broadway.eda.quality import QualityAnalyzer
from broadway.eda.report import EdaReportBuilder
from broadway.eda.segments import SegmentPlan
from broadway.eda.summary import SummaryAnalyzer
from broadway.eda.tests import FeatureTestAnalyzer
from broadway.eda.visualize import FigureRenderer
from broadway.observability.profiling import phase, timed

logger = logging.getLogger(__name__)

_PRE_SPLIT_LABEL = "canonical (pre-split)"
_POST_SPLIT_LABEL = "train+val (post-split)"
_HASH_LENGTH = 16
_READ_BATCH_ROWS = 250_000

_STREAM_TRAIN = 0
_STREAM_VAL = 1
_STREAM_CANONICAL = 2


class EdaRunner:
    """Runs the EDA analyses over the existing ETL split parquets (no ETL rerun)."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.dataset is None or cfg.eda is None or cfg.etl is None:
            raise ValueError("eda step requires dataset, eda, and etl config")
        self._cfg = cfg
        self._dataset: DatasetContract = cfg.dataset
        self._step: EdaStep = cfg.eda
        self._etl: EtlStep = cfg.etl
        self._random_state = cfg.eda.random_state
        self._source_rows = 0

    def run(self) -> Path:
        with timed() as elapsed:
            with phase("load"):
                profiled, source_rows, notes = self._load_split()
                before, before_rows, skip_reason = self._load_before()
                self._source_rows = source_rows
            with phase("analyze"):
                profile = self._profile_frame(profiled)
                plan = SegmentPlan.build(
                    profile, self._dataset.datetime_column, self._step.max_categories
                )
                summary = SummaryAnalyzer(self._step.max_categories).analyze(
                    profile, self._roles()
                )
                missing = MissingnessAnalyzer(self._step.max_categories).analyze(profile)
                quality = QualityAnalyzer(
                    self._step.iqr_multiplier,
                    self._step.z_threshold,
                    self._step.imbalance_min_share,
                ).analyze(profile, self._dataset.task, self._dataset.target)
                provenance = self._provenance(profile)
                compare = EtlComparator(
                    _PRE_SPLIT_LABEL, _POST_SPLIT_LABEL, self._step
                ).compare(
                    before,
                    profile,
                    before_rows,
                    source_rows,
                    skip_reason,
                    notes,
                    plan,
                    provenance,
                )
            with phase("decide"):
                correlation = CorrelationAnalyzer(self._step).analyze(
                    profile, self._dataset.target
                )
                feature_tests = FeatureTestAnalyzer(
                    self._step, self._dataset.task
                ).analyze(profile, self._dataset.target, provenance, plan)
                outliers = OutlierDecisionAnalyzer(self._step).analyze(
                    quality, profile, provenance, plan
                )
            with phase("figures"):
                figures = self._figures(
                    profile, summary, missing, correlation, feature_tests, outliers
                )
            with phase("report"):
                result = EdaResult(
                    dataset=self._dataset.name,
                    experiment=None,
                    task=self._dataset.task.value,
                    source_rows=source_rows,
                    sampled_rows=len(profiled),
                    sampled=len(profiled) < source_rows,
                    summary=summary,
                    missing=missing,
                    quality=quality,
                    compare=compare,
                    correlation=correlation,
                    feature_tests=feature_tests,
                    outliers=outliers,
                    figures=figures,
                )
                out_path = self._write(result)
        logger.info("eda: report written to %s in %.2fs", out_path, elapsed())
        logger.info(
            "eda: rows=%d sampled=%d missing_cells=%d constant=%d duplicates=%d "
            "outlier_columns=%d features_tested=%d decisions=%s compare=%s",
            source_rows,
            len(profile),
            missing.missing_cells,
            len(quality.constant_columns),
            quality.duplicate_rows,
            len(quality.outliers),
            len(feature_tests.features),
            self._verdict_counts(feature_tests),
            "available" if compare.available else f"skipped ({compare.skip_reason})",
        )
        return out_path

    def _figures(
        self,
        frame: pd.DataFrame,
        summary: SummaryResult,
        missing: MissingResult,
        correlation: CorrelationResult,
        feature_tests: FeatureTestBundle,
        outliers: OutlierResult,
    ) -> list[EmbeddedFigure]:
        """The descriptive figures plus one saved figure per decision metric."""
        renderer = FigureRenderer(
            self._step.plot_format, self._step.max_categories, self._step.max_plots
        )
        figures = renderer.render(frame, summary, missing)
        decision_figures = DecisionFigures()
        evidence: tuple[tuple[str, str, Figure | None], ...] = (
            (
                "effects",
                f"associations with {self._dataset.target}",
                decision_figures.effects(feature_tests),
            ),
            ("correlations", "numeric correlation", decision_figures.correlation(correlation)),
            ("outliers", "outlier share", decision_figures.outliers(outliers)),
        )
        writer = self._writer()
        for name, title, figure in evidence:
            if figure is None:
                continue
            writer.write_figure(name, figure, self._step.plot_format)
            figures.append(renderer.embed(figure, title))
        return figures

    def _writer(self) -> EdaArtifactWriter:
        return EdaArtifactWriter(self._step.output_path.parent, self._step.stem)

    def _write(self, result: EdaResult) -> Path:
        writer = self._writer()
        writer.prepare()
        writer.write_tables(
            result.feature_tests, result.correlation, result.outliers, result.compare
        )
        writer.write_decisions(
            EdaDecisionBundle(
                dataset=result.dataset,
                summary=result.feature_tests.summary,
                comparison=result.compare.decision,
                features=result.feature_tests.decisions,
                outliers=result.outliers.decisions,
            )
        )
        return writer.write_html(EdaReportBuilder().build(result))

    def _provenance(self, frame: pd.DataFrame) -> dict[str, str]:
        """The digest a reader needs to re-run these decisions on the same rows."""
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        ).hexdigest()[:_HASH_LENGTH]
        return {
            "artifact": f"{self._etl.train_file}+{self._etl.val_file}",
            "source": self._dataset.path,
            "sample_sha256": digest,
            "sampled_rows": str(len(frame)),
            "sample_size": str(self._step.sample_size or self._source_rows),
            "seed": str(self._random_state),
            "exclusions": "pairwise-complete rows per feature; exclude_from_profiling dropped",
        }

    @staticmethod
    def _verdict_counts(tests: FeatureTestBundle) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in tests.features:
            key = row.verdict or "undecided"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _load_split(self) -> tuple[pd.DataFrame, int, list[str]]:
        """One proportional draw of train+val (never 50k from each split)."""
        notes: list[str] = []
        out_dir = Path(self._cfg.environment.data_dir) / self._cfg.environment.processed_subdir
        train_path = out_dir / self._etl.train_file
        val_path = out_dir / self._etl.val_file
        train_rows = self._row_count(train_path, "train split")
        val_rows = self._row_count(val_path, "val split") if val_path.exists() else 0
        total = train_rows + val_rows
        budget = total if not self._step.sample_size else min(self._step.sample_size, total)
        train_n = budget if total == 0 else max(1, round(budget * train_rows / total))
        train_n = min(train_n, train_rows)
        val_n = min(budget - train_n, val_rows)
        train_sample = self._read_sample(train_path, "train split", train_n, _STREAM_TRAIN)
        parts = [train_sample]
        if val_n > 0:
            parts.append(self._read_sample(val_path, "val split", val_n, _STREAM_VAL))
            notes.append(
                f"after-ETL frame is one proportional draw: train {train_n:,} of "
                f"{train_rows:,} + val {val_n:,} of {val_rows:,} rows"
            )
        else:
            if not val_path.exists():
                notes.append(f"val split not found: {val_path} — profiling train split only")
            else:
                notes.append(
                    f"sample budget {budget:,} left no rows for the val split; "
                    "profiling the train draw only"
                )
        return pd.concat(parts, ignore_index=True), total, notes

    def _load_before(self) -> tuple[pd.DataFrame | None, int, str | None]:
        path = canonical_path(self._dataset, self._cfg.environment)
        if not path.exists():
            return (
                None,
                0,
                f"pre-split canonical frame not found: {path} — run the etl step first",
            )
        total = self._row_count(path, "canonical frame")
        budget = total if not self._step.sample_size else min(self._step.sample_size, total)
        return (
            self._read_sample(path, "canonical frame", budget, _STREAM_CANONICAL),
            total,
            None,
        )

    @staticmethod
    def _row_count(path: Path, label: str) -> int:
        if not path.exists():
            raise FileNotFoundError(f"eda: {label} not found: {path}")
        return int(pq.ParquetFile(path).metadata.num_rows)

    def _read_sample(self, path: Path, label: str, count: int, stream: int) -> pd.DataFrame:
        """Read ``count`` seeded rows in bounded batches (never the whole frame).

        The target row positions are drawn once from the file's row count, then
        each parquet batch contributes only the rows whose positions fall in its
        range — so peak memory is one batch plus the sample.
        """
        total = self._row_count(path, label)
        logger.info(
            "eda: reading %s from %s (%d of %d rows, stream %d)",
            label,
            path,
            count,
            total,
            stream,
        )
        if count >= total:
            return pd.read_parquet(path)
        wanted = np.sort(
            np.random.default_rng([self._random_state, stream]).choice(
                total, size=count, replace=False
            )
        )
        parts: list[pd.DataFrame] = []
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=_READ_BATCH_ROWS):
            size = batch.num_rows
            local = wanted[(wanted >= offset) & (wanted < offset + size)] - offset
            if local.size:
                parts.append(batch.take(pa.array(local)).to_pandas())
            offset += size
        if not parts:
            raise ValueError(f"eda: the {label} draw selected no rows from {path}")
        return pd.concat(parts, ignore_index=True)

    def _profile_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        excluded = [
            name
            for name in self._dataset.exclude_from_profiling
            if name in frame.columns
        ]
        return frame.drop(columns=excluded) if excluded else frame

    def _roles(self) -> dict[str, str]:
        return {
            name: schema.role.value
            for name, schema in self._dataset.columns.items()
        }


def run(cfg: PipelineConfig) -> None:
    """Pipeline entry point; records the full traceback and re-raises on failure."""
    try:
        EdaRunner(cfg).run()
    except Exception:
        logger.exception("eda step failed")
        raise
