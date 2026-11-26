"""EDA pipeline step — orchestrates summary, missingness, quality, decisions and report."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from broadway.config.schema import DatasetContract, EdaStep, EtlStep, PipelineConfig
from broadway.data.loader import canonical_path
from broadway.eda.artifacts import EdaArtifactWriter
from broadway.eda.compare import EtlComparator
from broadway.eda.correlation import CorrelationAnalyzer
from broadway.eda.figures import DecisionFigures
from broadway.eda.missing import MissingnessAnalyzer
from broadway.eda.models import (
    CompareResult,
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
from broadway.eda.summary import SummaryAnalyzer
from broadway.eda.tests import FeatureTestAnalyzer
from broadway.eda.visualize import FigureRenderer
from broadway.observability.profiling import phase, timed

logger = logging.getLogger(__name__)

_PRE_SPLIT_LABEL = "canonical (pre-split)"
_POST_SPLIT_LABEL = "train+val (post-split)"
_HASH_LENGTH = 16


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
                after, source_rows, notes = self._load_split()
                before, before_rows, skip_reason = self._load_before()
                self._source_rows = source_rows
            with phase("sample"):
                profiled = self._sample(after)
                before_sample = self._sample(before) if before is not None else None
            with phase("analyze"):
                profile = self._profile_frame(profiled)
                summary = SummaryAnalyzer(self._step.max_categories).analyze(
                    profile, self._roles()
                )
                missing = MissingnessAnalyzer(self._step.max_categories).analyze(profile)
                quality = QualityAnalyzer(
                    self._step.iqr_multiplier,
                    self._step.z_threshold,
                    self._step.imbalance_min_share,
                ).analyze(profile, self._dataset.task, self._dataset.target)
                compare = self._compare(
                    before_sample, profiled, before_rows, source_rows, skip_reason, notes
                )
            with phase("decide"):
                provenance = self._provenance(profile)
                correlation = CorrelationAnalyzer(self._step).analyze(
                    profile, self._dataset.target
                )
                feature_tests = FeatureTestAnalyzer(self._step, self._dataset.task).analyze(
                    profile, self._dataset.target, provenance
                )
                outliers = OutlierDecisionAnalyzer(self._step).analyze(
                    quality, profile, provenance
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
            "eda: rows=%d profiled=%d missing_cells=%d constant=%d duplicates=%d "
            "outlier_columns=%d features_tested=%d decisions=%s compare=%s",
            source_rows,
            len(profiled),
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
            ("effects", f"associations with {self._dataset.target}", decision_figures.effects(feature_tests)),
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
        writer.write_tables(result.feature_tests, result.correlation, result.outliers)
        writer.write_decisions(
            EdaDecisionBundle(
                dataset=result.dataset,
                summary=result.feature_tests.summary,
                features=result.feature_tests.decisions,
                outliers=result.outliers.decisions,
            )
        )
        return writer.write_html(EdaReportBuilder().build(result))

    def _provenance(self, frame: pd.DataFrame) -> dict[str, str]:
        """The digest a reader needs to re-run this decision on the same rows."""
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        ).hexdigest()[:_HASH_LENGTH]
        return {
            "artifact": f"{self._etl.train_file}+{self._etl.val_file}",
            "source": self._dataset.path,
            "sample_sha256": digest,
            "sample_size": str(self._step.sample_size),
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
        notes: list[str] = []
        out_dir = Path(self._cfg.environment.data_dir) / self._cfg.environment.processed_subdir
        train_path = out_dir / self._etl.train_file
        train = self._read(train_path, "train split")
        train_rows = len(train)
        train_sample = self._sample(train)
        del train
        val_path = out_dir / self._etl.val_file
        if val_path.exists():
            val = self._read(val_path, "val split")
            val_rows = len(val)
            val_sample = self._sample(val)
            del val
            notes.append(
                f"after-ETL frame combines train ({train_rows}) + val ({val_rows}) rows"
            )
            return (
                pd.concat([train_sample, val_sample], ignore_index=True),
                train_rows + val_rows,
                notes,
            )
        notes.append(f"val split not found: {val_path} — profiling train split only")
        return train_sample, train_rows, notes

    def _load_before(self) -> tuple[pd.DataFrame | None, int, str | None]:
        path = canonical_path(self._dataset, self._cfg.environment)
        if not path.exists():
            return (
                None,
                0,
                f"pre-split canonical frame not found: {path} — run the etl step first",
            )
        before = self._read(path, "canonical frame")
        return before, len(before), None

    def _read(self, path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"eda: {label} not found: {path}")
        logger.info("eda: reading %s from %s", label, path)
        return pd.read_parquet(path)

    def _sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._step.sample_size and len(frame) > self._step.sample_size:
            return frame.sample(
                n=self._step.sample_size, random_state=self._random_state
            ).reset_index(drop=True)
        return frame.reset_index(drop=True)

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

    def _compare(
        self,
        before: pd.DataFrame | None,
        after: pd.DataFrame,
        before_rows: int,
        after_rows: int,
        skip_reason: str | None,
        notes: list[str],
    ) -> CompareResult:
        return EtlComparator(_PRE_SPLIT_LABEL, _POST_SPLIT_LABEL).compare(
            before,
            after,
            before_rows,
            after_rows,
            skip_reason,
            notes,
        )


def run(cfg: PipelineConfig) -> None:
    """Pipeline entry point; records the full traceback and re-raises on failure."""
    try:
        EdaRunner(cfg).run()
    except Exception:
        logger.exception("eda step failed")
        raise
