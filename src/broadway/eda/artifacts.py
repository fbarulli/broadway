"""The EDA step's artifacts on disk: report, evidence tables, figures, decisions.

One writer, one stem. ``<stem>.html`` is the human report; ``<stem>_effects.csv``,
``<stem>_correlations.csv``, ``<stem>_outliers.csv`` and ``<stem>_comparison.csv``
are the aggregate evidence tables rendered under the figures;
``<stem>_decision.json`` is **the** decision record — singular, the summary
metric the dashboard reads (``experiments_dashboard`` looks for
``<stem>_decision.json`` and ``.gitignore`` un-ignores only that name).
``<stem>_feature_decisions.json`` is the per-feature evidence written next to
it, carrying each family's decision. ``<stem>_<figure>.<format>`` are the three
figures. Aggregate rows only — a CSV never carries a raw datapoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from broadway.config.viz import load_viz_config
from broadway.eda.models import (
    CompareResult,
    CorrelationResult,
    EdaDecisionBundle,
    FeatureTestBundle,
    OutlierResult,
)

logger = logging.getLogger(__name__)

_DECIMALS = 6


class EdaArtifactWriter:
    """Writes every EDA artifact under one output directory and stem."""

    def __init__(self, output_dir: Path, stem: str) -> None:
        self._directory = Path(output_dir)
        self._stem = stem
        self._dpi = load_viz_config().dpi

    @property
    def directory(self) -> Path:
        return self._directory

    def html_path(self) -> Path:
        return self._directory / f"{self._stem}.html"

    def decision_path(self) -> Path:
        """The singular record the dashboard reads (``<stem>_decision.json``)."""
        return self._directory / f"{self._stem}_decision.json"

    def feature_decisions_path(self) -> Path:
        """Every family's decisions as evidence under the singular record."""
        return self._directory / f"{self._stem}_feature_decisions.json"

    def comparison_path(self) -> Path:
        return self._directory / f"{self._stem}_comparison.csv"

    def effects_path(self) -> Path:
        return self._directory / f"{self._stem}_effects.csv"

    def correlations_path(self) -> Path:
        return self._directory / f"{self._stem}_correlations.csv"

    def outliers_path(self) -> Path:
        return self._directory / f"{self._stem}_outliers.csv"

    def figure_path(self, name: str, plot_format: str) -> Path:
        return self._directory / f"{self._stem}_{name}.{plot_format}"

    def prepare(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

    def write_html(self, html: str) -> Path:
        """Write the self-contained report atomically (a rerun never tears it)."""
        self.prepare()
        path = self.html_path()
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(path)
        return path

    def write_figure(self, name: str, figure: Figure, plot_format: str) -> Path:
        self.prepare()
        path = self.figure_path(name, plot_format)
        figure.savefig(path, format=plot_format, dpi=self._dpi)
        return path

    def remove_figure(self, name: str, plot_format: str) -> Path | None:
        """Delete a stale figure of this run's stem; a skipped panel leaves no file.

        A figure the current run cannot draw (e.g. no numeric column pair) must not
        survive from an earlier run beside today's record and report: the file name
        is the step's, not the run's, so a leftover is read as current evidence.
        """
        path = self.figure_path(name, plot_format)
        if not path.exists():
            return None
        path.unlink()
        logger.info("eda: removed stale figure %s (this run drew none)", path)
        return path

    def write_tables(
        self,
        tests: FeatureTestBundle,
        correlation: CorrelationResult,
        outliers: OutlierResult,
        comparison: CompareResult,
    ) -> list[Path]:
        self.prepare()
        return [
            self._write(self.effects_path(), self._effects_frame(tests)),
            self._write(self.correlations_path(), self._correlations_frame(correlation)),
            self._write(self.outliers_path(), self._outliers_frame(outliers)),
            self._write(self.comparison_path(), self._comparison_frame(comparison)),
        ]

    def write_decisions(self, bundle: EdaDecisionBundle) -> list[Path]:
        """Write the singular record plus its per-feature evidence sidecar."""
        self.prepare()
        sidecar = self.feature_decisions_path()
        sidecar.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        if bundle.summary is None:
            logger.warning(
                "eda: no summary decision to publish — every family was unmeasurable; "
                "the report records why (evidence: %s)",
                sidecar,
            )
            return [sidecar]
        path = self.decision_path()
        path.write_text(bundle.summary.model_dump_json(indent=2), encoding="utf-8")
        return [path, sidecar]

    @staticmethod
    def _write(path: Path, frame: pd.DataFrame) -> Path:
        frame.to_csv(path, index=False)
        return path

    @staticmethod
    def _effects_frame(tests: FeatureTestBundle) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": [row.feature for row in tests.features],
                "feature_kind": [row.feature_kind for row in tests.features],
                "target_kind": [row.target_kind for row in tests.features],
                "test": [row.test for row in tests.features],
                "effect_name": [row.effect_name for row in tests.features],
                "effect": [round(row.effect, _DECIMALS) for row in tests.features],
                "p_value": [round(row.p_value, _DECIMALS) for row in tests.features],
                "q_value": [round(row.q_value, _DECIMALS) for row in tests.features],
                "significant": [row.significant for row in tests.features],
                "n": [row.n for row in tests.features],
                "alpha": [tests.alpha for _ in tests.features],
                "min_effect": [tests.min_effect for _ in tests.features],
                "verdict": [row.verdict or "" for row in tests.features],
            }
        )

    def _correlations_frame(self, correlation: CorrelationResult) -> pd.DataFrame:
        rows = correlation.pairs
        return pd.DataFrame(
            {
                "left": [pair.left for pair in rows],
                "right": [pair.right for pair in rows],
                "n": [pair.n for pair in rows],
                "pearson": [self._rounded(pair.pearson) for pair in rows],
                "pearson_p": [self._rounded(pair.pearson_p) for pair in rows],
                "pearson_q": [self._rounded(pair.pearson_q) for pair in rows],
                "spearman": [self._rounded(pair.spearman) for pair in rows],
                "spearman_p": [self._rounded(pair.spearman_p) for pair in rows],
                "spearman_q": [self._rounded(pair.spearman_q) for pair in rows],
            }
        )

    def _comparison_frame(self, comparison: CompareResult) -> pd.DataFrame:
        rows = comparison.numeric_shifts
        return pd.DataFrame(
            {
                "column": [item.column for item in rows],
                "before_mean": [self._rounded(item.before_mean) for item in rows],
                "after_mean": [self._rounded(item.after_mean) for item in rows],
                "mean_delta": [self._rounded(item.mean_delta) for item in rows],
                "mean_delta_low": [self._rounded(item.mean_delta_low) for item in rows],
                "mean_delta_high": [self._rounded(item.mean_delta_high) for item in rows],
                "median_delta": [self._rounded(item.median_delta) for item in rows],
                "median_delta_low": [self._rounded(item.median_delta_low) for item in rows],
                "median_delta_high": [self._rounded(item.median_delta_high) for item in rows],
            }
        )

    @staticmethod
    def _outliers_frame(outliers: OutlierResult) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": [item.column for item in outliers.columns],
                "n": [item.n for item in outliers.columns],
                "iqr_count": [item.iqr_count for item in outliers.columns],
                "iqr_share": [round(item.iqr_share, _DECIMALS) for item in outliers.columns],
                "z_count": [item.z_count for item in outliers.columns],
                "z_share": [round(item.z_share, _DECIMALS) for item in outliers.columns],
                "max_share": [item.max_share for item in outliers.columns],
                "verdict": [item.verdict for item in outliers.columns],
            }
        )


    @staticmethod
    def _rounded(value: float | None) -> float | None:
        return None if value is None else round(value, _DECIMALS)
