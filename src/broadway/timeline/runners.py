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
from broadway.stats.anova import run_anova, run_kruskal, run_welch
from broadway.stats.assumptions import check_normality, run_levene
from broadway.stats.describe import describe
from broadway.stats.post_hoc import games_howell
from broadway.timeline.evidence import (
    ConclusionEvidence,
    NormalityEvidence,
    NormalityGroupStat,
    PosthocEvidence,
    PosthocPair,
    VarianceEvidence,
)
from broadway.timeline.models import AnalysisStep, StepStatus
from broadway.timeline.sequence import WalkthroughConfig, load_walkthrough_config


def _thresholds() -> WalkthroughConfig:
    return load_walkthrough_config()


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
    thresholds = _thresholds()
    flagged = summary.imbalance_ratio > thresholds.imbalance_ratio_threshold or bool(summary.absent_groups)
    if flagged:
        parts = []
        if summary.absent_groups:
            parts.append(f"groups {', '.join(summary.absent_groups)} have no observations")
        if summary.imbalance_ratio > thresholds.imbalance_ratio_threshold:
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
    thresholds = _thresholds()
    flagged = any(
        abs(s["skew"]) > thresholds.skew_threshold
        or abs(s["kurtosis"]) > thresholds.kurtosis_threshold
        or s["shapiro_p"] < thresholds.shapiro_alpha
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
    thresholds = _thresholds()
    flagged = result["p_value"] < thresholds.shapiro_alpha
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


def run_omnibus(
    analysis: AnalysisContract,
    order: int,
    question: str,
    groups: dict[str, np.ndarray],
    method: str,
    out_dir: Path,
    source: str,
    sample_name: str | None,
) -> AnalysisStep:
    if method == "anova":
        plan = run_anova(groups)
    elif method == "welch":
        plan = run_welch(groups)
    elif method == "kruskal":
        plan = run_kruskal(groups)
    else:
        raise ValueError(f"unknown omnibus method '{method}'")
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "omnibus.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    p_value = plan.statistics["p_value"]
    if plan.passed:
        ramification = f"reject H0: at least one group mean differs (p={p_value:.4e})"
    else:
        ramification = f"fail to reject H0: no mean difference (p={p_value:.4e})"
    if method == "kruskal":
        ramification += (
            " rank-based effect size not yet implemented; Broadway did not "
            "substitute an ANOVA effect-size measure."
        )
    if plan.reason:
        ramification += " " + "; ".join(plan.reason)
    result_summary: dict[str, object] = {
        "method": method,
        "statistic": plan.statistics["statistic"],
        "p_value": p_value,
        "passed": plan.passed,
    }
    if method in ("anova", "welch"):
        result_summary["eta_squared"] = plan.effect_sizes["eta_squared"]
        result_summary["omega_squared"] = plan.effect_sizes["omega_squared"]
    else:
        result_summary["effect_size"] = "not_available"
    return AnalysisStep(
        analysis=analysis.name,
        step_id="omnibus",
        order=order,
        question=question,
        status=StepStatus.WARNING if plan.warnings else StepStatus.COMPLETED,
        method=method,
        source=source,
        sample_name=sample_name,
        evidence_refs=["omnibus.json"],
        result_summary=result_summary,
        ramification=ramification,
        decision_required=plan.passed,
        performed_at=now_iso(),
    )


def run_posthoc(
    analysis: AnalysisContract,
    order: int,
    question: str,
    df: pd.DataFrame,
    source_group_column: str,
    target: str,
    method: str,
    out_dir: Path,
    source: str,
    sample_name: str | None,
) -> AnalysisStep:
    if method != "games_howell":
        raise ValueError(f"unknown posthoc method '{method}'")
    result = games_howell(df, dv=target, between=source_group_column)
    pairs = [
        PosthocPair(
            a=row["A"],
            b=row["B"],
            p_value=float(row["pval"]),
            cohens_d=float(row["cohens_d"]),
            hedges_g=float(row["hedges_g"]),
            effect_size_note=row["effect_size_note"],
        )
        for _, row in result.iterrows()
    ]
    thresholds = _thresholds()
    significant_pairs = sum(1 for p in pairs if p.p_value < thresholds.significance_alpha)
    evidence = PosthocEvidence(method=method, pairs=pairs, significant_pairs=significant_pairs)
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "posthoc.json").write_text(
        evidence.model_dump_json(indent=2), encoding="utf-8"
    )
    return AnalysisStep(
        analysis=analysis.name,
        step_id="posthoc",
        order=order,
        question=question,
        status=StepStatus.COMPLETED,
        method=method,
        source=source,
        sample_name=sample_name,
        evidence_refs=["posthoc.json"],
        result_summary={
            "method": method,
            "pairs": len(pairs),
            "significant_pairs": significant_pairs,
        },
        ramification=(
            f"Games-Howell found {significant_pairs} significant pairwise "
            "difference(s) at alpha=0.05."
        ),
        decision_required=False,
        performed_at=now_iso(),
    )


def run_conclusion(
    analysis: AnalysisContract,
    order: int,
    question: str,
    omnibus_step: AnalysisStep,
    posthoc_step: AnalysisStep | None,
    out_dir: Path,
    source: str,
    sample_name: str | None,
) -> AnalysisStep:
    summary = omnibus_step.result_summary
    method = str(summary["method"])
    p_value = float(summary["p_value"])
    passed = bool(summary["passed"])
    if method in ("anova", "welch"):
        effect_size = (
            f"eta²={float(summary['eta_squared']):.4g}, "
            f"omega²={float(summary['omega_squared']):.4g}"
        )
    else:
        effect_size = "not available (rank-based)"
    significant_pairs = (
        int(posthoc_step.result_summary.get("significant_pairs", 0))
        if posthoc_step is not None
        else 0
    )
    if passed:
        verdict = (
            f"group means differ ({method} p={p_value:.4e}), with {significant_pairs} "
            "significant pairwise difference(s)."
        )
    else:
        verdict = f"no significant difference across groups ({method} p={p_value:.4e})."
    notes = [omnibus_step.ramification]
    if posthoc_step is not None:
        notes.append(posthoc_step.ramification)
    evidence = ConclusionEvidence(
        verdict=verdict,
        principal_method=method,
        p_value=p_value,
        effect_size=effect_size,
        significant_pairs=significant_pairs,
        notes=notes,
    )
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "conclusion.json").write_text(
        evidence.model_dump_json(indent=2), encoding="utf-8"
    )
    return AnalysisStep(
        analysis=analysis.name,
        step_id="conclusion",
        order=order,
        question=question,
        status=StepStatus.COMPLETED,
        method="conclusion",
        source=source,
        sample_name=sample_name,
        evidence_refs=["conclusion.json"],
        result_summary={
            "verdict": verdict,
            "principal_method": method,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant_pairs": significant_pairs,
        },
        ramification=verdict,
        decision_required=False,
        performed_at=now_iso(),
    )
