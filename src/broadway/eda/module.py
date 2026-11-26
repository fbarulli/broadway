"""EDA pipeline step — orchestrates summary, missingness, quality and report."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from broadway.config.schema import DatasetContract, EdaStep, EtlStep, PipelineConfig
from broadway.data.loader import canonical_path
from broadway.eda.compare import EtlComparator
from broadway.eda.missing import MissingnessAnalyzer
from broadway.eda.models import CompareResult, EdaResult
from broadway.eda.quality import QualityAnalyzer
from broadway.eda.report import EdaReportBuilder
from broadway.eda.summary import SummaryAnalyzer
from broadway.eda.visualize import FigureRenderer
from broadway.observability.profiling import phase, timed

logger = logging.getLogger(__name__)

_PRE_SPLIT_LABEL = "canonical (pre-split)"
_POST_SPLIT_LABEL = "train+val (post-split)"


class EdaRunner:
    """Runs the EDA analyses over the existing ETL split parquets (no ETL rerun)."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.dataset is None or cfg.eda is None or cfg.etl is None:
            raise ValueError("eda step requires dataset, eda, and etl config")
        self._cfg = cfg
        self._dataset: DatasetContract = cfg.dataset
        self._step: EdaStep = cfg.eda
        self._etl: EtlStep = cfg.etl
        self._random_state = (
            cfg.experiment.random_state if cfg.experiment else cfg.etl.random_state
        )

    def run(self) -> Path:
        with timed() as elapsed:
            with phase("load"):
                after, source_rows, notes = self._load_split()
                before, before_rows, skip_reason = self._load_before()
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
            with phase("figures"):
                figures = FigureRenderer(
                    self._step.plot_format,
                    self._step.max_categories,
                    self._step.max_plots,
                ).render(profile, summary, missing, quality)
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
                    figures=figures,
                )
                out_path = self._write(result)
        logger.info("eda: report written to %s in %.2fs", out_path, elapsed())
        logger.info(
            "eda: rows=%d profiled=%d missing_cells=%d constant=%d duplicates=%d "
            "outlier_columns=%d compare=%s",
            source_rows,
            len(profiled),
            missing.missing_cells,
            len(quality.constant_columns),
            quality.duplicate_rows,
            len(quality.outliers),
            "available" if compare.available else f"skipped ({compare.skip_reason})",
        )
        return out_path

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

    def _write(self, result: EdaResult) -> Path:
        out_path = self._step.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = EdaReportBuilder().build(result)
        temporary = out_path.with_name(f"{out_path.name}.tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(out_path)
        return out_path


def run(cfg: PipelineConfig) -> None:
    """Pipeline entry point; records the full traceback and re-raises on failure."""
    try:
        EdaRunner(cfg).run()
    except Exception:
        logger.exception("eda step failed")
        raise
