"""Does a weekly-seasonal forecast beat a same-weekday seasonal-naive baseline?

The lane reads the canonical order-grain table through the dataset contract,
builds one daily series per configured spec (order count and revenue for
Olist), and scores a seasonal-naive baseline first, then a weekly-seasonal
AutoARIMA/SARIMA, on identical chronological folds. It publishes a canonical
Decision record per series, a CV table, figures and the aggregate report.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd

from broadway.config.schema import PipelineConfig, SeriesSpec, TimeseriesStep
from broadway.data.loader import canonical_path
from broadway.reports.decision import Decision
from broadway.reports.decision_log import DecisionLog
from broadway.reports.step_output import StepOutput
from broadway.timeseries.analysis import SeriesDecisionBuilder
from broadway.timeseries.backtest import BacktestResult, ForecastBacktest, SeasonalNaive
from broadway.timeseries.figures import TimeseriesFigures
from broadway.timeseries.model import ForecastBackend
from broadway.timeseries.report import TimeseriesReport, cv_frame
from broadway.timeseries.series import DailySeries, DailySeriesBuilder

logger = logging.getLogger(__name__)

STEP_NAME = "08_timeseries"
_HASH_LENGTH = 16


class TimeseriesRunner:
    """Run the configured daily-series forecast lane and publish its evidence."""

    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.dataset is None or cfg.timeseries is None:
            raise ValueError("08_timeseries step requires dataset and timeseries config")
        self._cfg = cfg
        self._step: TimeseriesStep = cfg.timeseries

    def run(self) -> Path:
        """Build the series, backtest, decide, and write record/table/figure/report."""
        started = time.perf_counter()
        frame = self._read_canonical()
        builder = DailySeriesBuilder.from_config(self._step)
        series = {
            name: builder.build(frame, name, spec)
            for name, spec in self._step.series.items()
        }
        backend = ForecastBackend.from_config(self._step)
        backtest = ForecastBacktest(
            SeasonalNaive(self._step.season_length), backend, self._step.cv
        )
        results = {name: backtest.run(daily) for name, daily in series.items()}
        decisions = self._decisions(series, results, backend.name)
        report = self._write(series, results, decisions, backend.name)
        logger.info(
            "timeseries: %s published at %s in %.2fs (backend=%s, folds=%d)",
            STEP_NAME,
            report,
            time.perf_counter() - started,
            backend.name,
            self._step.cv.folds,
        )
        return report

    def _read_canonical(self) -> pd.DataFrame:
        step = self._step
        assert self._cfg.dataset is not None
        path = canonical_path(self._cfg.dataset, self._cfg.environment)
        if not path.is_file():
            raise FileNotFoundError(
                f"08_timeseries: canonical table not found at {path}; run the etl step first"
            )
        columns = [step.datetime_column, *[
            spec.column for spec in step.series.values() if spec.column is not None
        ]]
        unique = list(dict.fromkeys(columns))
        logger.info("timeseries: reading %s columns=%s", path, unique)
        return pd.read_parquet(path, columns=unique)

    def _decisions(
        self,
        series: dict[str, DailySeries],
        results: dict[str, BacktestResult],
        backend: str,
    ) -> dict[str, Decision]:
        step = self._step
        builder = SeriesDecisionBuilder(step.rule, step.bootstrap, step.permutation)
        decisions: dict[str, Decision] = {}
        for name, daily in series.items():
            decisions[name] = builder.build(
                result=results[name],
                question=self._question(step.series[name]),
                provenance=self._provenance(daily, results[name], backend),
            )
        return decisions

    @staticmethod
    def _question(spec: SeriesSpec) -> str:
        return (
            "Does a weekly-seasonal forecast beat a same-weekday seasonal-naive "
            f"baseline for {spec.label}?"
        )

    def _provenance(
        self,
        series: DailySeries,
        result: BacktestResult,
        backend: str,
    ) -> dict[str, str]:
        step = self._step
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(series.frame(), index=False).to_numpy().tobytes()
        ).hexdigest()[:_HASH_LENGTH]
        return {
            "dataset": self._cfg.dataset.name if self._cfg.dataset else "",
            "artifact": f"{self._cfg.dataset.name if self._cfg.dataset else ''}_canonical.parquet",
            "series": series.name,
            "aggregation": series.aggregation,
            "column": series.column or "",
            "datetime_column": step.datetime_column,
            "grain_column": step.grain_column,
            "window": f"{step.start}..{step.end}",
            "share_sha256": digest,
            "model_backend": backend,
            "seed": str(step.bootstrap.seed),
            "folds": str(len(result.folds)),
            "exclusions": step.window_note or "none",
        }

    def _write(
        self,
        series: dict[str, DailySeries],
        results: dict[str, BacktestResult],
        decisions: dict[str, Decision],
        backend: str,
    ) -> Path:
        step = self._step
        target = step.output_path
        target.mkdir(parents=True, exist_ok=True)
        table = cv_frame(results)
        figures = TimeseriesFigures(series, results)
        log = DecisionLog(tracking_uri=self._cfg.environment.mlflow_tracking_uri)
        primary = step.primary_series
        primary_output = StepOutput(
            results_dir=target,
            stem=STEP_NAME,
            decision=decisions[primary],
            log=log,
        )
        primary_output.save_figure(figures.render(list(series)))
        primary_output.add_table("cv", table)
        primary_output.write_decision()
        for name in series:
            if name == primary:
                continue
            secondary = StepOutput(
                results_dir=target,
                stem=f"{STEP_NAME}_{name}",
                decision=decisions[name],
                log=log,
            )
            secondary.save_figure(figures.render([name]))
            secondary.write_decision()
        report = TimeseriesReport(
            STEP_NAME, step, results, decisions, backend
        ).render()
        report_path = target / f"{STEP_NAME}.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path


def run(cfg: PipelineConfig) -> None:
    """Pipeline entry point; records the full traceback and re-raises on failure."""
    try:
        TimeseriesRunner(cfg).run()
    except Exception:
        logger.exception("08_timeseries step failed")
        raise
