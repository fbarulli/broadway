from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from broadway.analysis.contracts import AnalysisContract
from broadway.config.schema import PipelineConfig
from broadway.data.loader import canonical_path
from broadway.lineage.models import SampleSpec
from broadway.stats.assumptions import check_normality, run_levene
from broadway.stats.describe import describe
from broadway.timeline.evidence import (
    NormalityEvidence,
    NormalityGroupStat,
    VarianceEvidence,
)
from broadway.timeline.models import AnalysisStep, StepStatus

_SKEW_THRESHOLD = 2.0
_KURTOSIS_THRESHOLD = 7.0
_SHAPIRO_ALPHA = 0.05
_IMBALANCE_THRESHOLD = 1.5


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "group"


def load_frame_and_groups(
    cfg: PipelineConfig, sample: SampleSpec | None
) -> tuple[pd.DataFrame, str, str, dict[str, np.ndarray]]:
    if cfg.dataset is None or cfg.analysis is None or cfg.analysis.hypothesis is None:
        raise ValueError("walkthrough requires dataset and hypothesis config")
    group_column = cfg.analysis.hypothesis.group_column
    group_values = cfg.analysis.hypothesis.group_values
    if sample is None:
        source_group_column = group_column
        path = canonical_path(cfg.dataset, cfg.environment)
        if not path.exists():
            raise FileNotFoundError(f"canonical dataset not found: {path} — run the etl step first")
        df = pd.read_parquet(path)
    else:
        source_group_column = sample.column_mapping.get(group_column, group_column)
        path = Path(sample.path)
        if not path.exists():
            raise FileNotFoundError(f"sample dataset not found: {path}")
        df = pd.read_parquet(path)
    if source_group_column not in df.columns:
        raise ValueError(f"group column '{source_group_column}' not found in data")
    groups = {
        g: df[df[source_group_column] == g][cfg.dataset.target].dropna().to_numpy()
        for g in group_values
    }
    return df, group_column, source_group_column, groups


def run_describe(
    analysis: AnalysisContract,
    order: int,
    question: str,
    df: pd.DataFrame,
    group_column: str,
    source_group_column: str,
    group_values: list[str],
    target: str,
    source_path: str,
    sample_name: str | None,
    source: str,
    out_dir: Path,
) -> AnalysisStep:
    summary = describe(
        df, group_column, source_group_column, group_values, target,
        source_path, sample_name or "canonical", "diagnostic",
    )
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "describe.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    flagged = summary.imbalance_ratio > _IMBALANCE_THRESHOLD or bool(summary.absent_groups)
    if flagged:
        parts = []
        if summary.absent_groups:
            parts.append(f"groups {', '.join(summary.absent_groups)} have no observations")
        if summary.imbalance_ratio > _IMBALANCE_THRESHOLD:
            parts.append(f"group sizes are imbalanced (imbalance ratio {summary.imbalance_ratio})")
        ramification = "; ".join(parts) + "."
    else:
        ramification = "group sizes are adequate."
    return AnalysisStep(
        analysis=analysis.name,
        step_id="describe_groups",
        order=order,
        question=question,
        status=StepStatus.WARNING if flagged else StepStatus.COMPLETED,
        method="describe",
        source=source,
        sample_name=sample_name,
        evidence_refs=["describe.json"],
        result_summary={
            "total_n": summary.total_n,
            "imbalance_ratio": summary.imbalance_ratio,
            "absent_groups": len(summary.absent_groups),
        },
        ramification=ramification,
        decision_required=False,
        performed_at=now_iso(),
    )


def _plot_qq(vals: np.ndarray, name: str, out_path: Path) -> None:
    arr = np.asarray(vals, dtype=float)
    (osm, osr), (slope, intercept, _r) = stats.probplot(arr, dist="norm", fit=True)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(osm, osr, s=10)
    ax.plot(osm, slope * osm + intercept, color="red")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(f"Q-Q plot — {name}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_normality(
    analysis: AnalysisContract,
    order: int,
    question: str,
    groups: dict[str, np.ndarray],
    out_dir: Path,
    figures_dir: Path,
    source: str,
    sample_name: str | None,
) -> AnalysisStep:
    result = check_normality(groups)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    for name, vals in groups.items():
        if len(vals) < 2:
            continue
        fname = f"normality_{_safe_filename(name)}.png"
        _plot_qq(vals, name, figures_dir / fname)
        figure_paths.append(f"figures/{fname}")
    evidence = NormalityEvidence(
        groups={g: NormalityGroupStat(**s) for g, s in result.items()},
        figures=figure_paths,
    )
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "normality.json").write_text(
        evidence.model_dump_json(indent=2), encoding="utf-8"
    )
    flagged = any(
        abs(s["skew"]) > _SKEW_THRESHOLD
        or abs(s["kurtosis"]) > _KURTOSIS_THRESHOLD
        or s["shapiro_p"] < _SHAPIRO_ALPHA
        for s in result.values()
    )
    ramification = (
        "distributional shape is skewed/heavy-tailed in some groups; consider this "
        "alongside sample size before choosing a method."
        if flagged
        else "distributional shape is broadly reasonable."
    )
    return AnalysisStep(
        analysis=analysis.name,
        step_id="normality",
        order=order,
        question=question,
        status=StepStatus.WARNING if flagged else StepStatus.COMPLETED,
        method="check_normality",
        source=source,
        sample_name=sample_name,
        evidence_refs=["normality.json"] + figure_paths,
        result_summary={
            g: {"skew": s["skew"], "kurtosis": s["kurtosis"], "shapiro_p": s["shapiro_p"]}
            for g, s in result.items()
        },
        ramification=ramification,
        decision_required=False,
        performed_at=now_iso(),
    )


def run_variance(
    analysis: AnalysisContract,
    order: int,
    question: str,
    groups: dict[str, np.ndarray],
    out_dir: Path,
    source: str,
    sample_name: str | None,
) -> AnalysisStep:
    result = run_levene(groups)
    evidence = VarianceEvidence(statistic=result["statistic"], p_value=result["p_value"])
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "variance.json").write_text(
        evidence.model_dump_json(indent=2), encoding="utf-8"
    )
    flagged = result["p_value"] < _SHAPIRO_ALPHA
    ramification = (
        "variance evidence favors considering Welch's ANOVA (or a rank-based alternative) "
        "over standard ANOVA."
        if flagged
        else "no evidence of unequal group variances."
    )
    return AnalysisStep(
        analysis=analysis.name,
        step_id="variance",
        order=order,
        question=question,
        status=StepStatus.WARNING if flagged else StepStatus.COMPLETED,
        method="levene",
        source=source,
        sample_name=sample_name,
        evidence_refs=["variance.json"],
        result_summary={"statistic": result["statistic"], "p_value": result["p_value"]},
        ramification=ramification,
        decision_required=False,
        performed_at=now_iso(),
    )
