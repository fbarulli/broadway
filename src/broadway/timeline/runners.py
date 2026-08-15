from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from broadway import viz
from broadway.analysis.contracts import AnalysisContract
from broadway.config.schema import PipelineConfig
from broadway.config.viz import load_viz_config
from broadway.data.loader import canonical_path
from broadway.lineage.models import SampleSpec
from broadway.reports.results import humanize_float, humanize_pvalue
from broadway.stats.anova import run_anova, run_kruskal, run_welch
from broadway.stats.assumptions import check_normality, run_levene
from broadway.stats.describe import describe, plot_describe_figures
from broadway.stats.post_hoc import games_howell
from broadway.timeline.evidence import (
    ConclusionEvidence,
    NormalityEvidence,
    NormalityGroupStat,
    PosthocEvidence,
    PosthocPair,
    VarianceEvidence,
)
from broadway.timeline.models import AnalysisStep, FigureRef, StepStatus
from broadway.timeline.sequence import WalkthroughConfig, load_walkthrough_config

BOXPLOT_CAPTION = (
    "How to read: the top panel is a boxplot of the target by group, each box spanning the "
    "interquartile range with a line at the median; the bottom panel shows the number of "
    "observations per group; very unequal bars indicate imbalance."
)
QQ_CAPTION = (
    "How to read: one plot per group; each group's standardized values are plotted "
    "against theoretical normal quantiles; points following the fitted reference line "
    "are approximately normal; curvature or heavy tails indicate non-normality."
)

logger = logging.getLogger(__name__)


def _thresholds() -> WalkthroughConfig:
    return load_walkthrough_config()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attrition(
    df: pd.DataFrame, source_group_column: str, group_values: list[str], target: str
) -> dict[str, int | str]:
    total = int(len(df))
    group_col = df[source_group_column]
    listed = group_col.isin(group_values)
    null_group = int(group_col.isna().sum())
    unlisted = int((group_col.notna() & ~listed).sum())
    null_target = int(df[target].isna().sum())
    used = int((listed & df[target].notna()).sum())
    reasons: list[str] = []
    if null_group:
        reasons.append("null group")
    if null_target:
        reasons.append("null target")
    if unlisted:
        reasons.append("unlisted group (values outside the configured group list; see audit)")
    return {
        "n_total": total,
        "n_used": used,
        "n_excluded": total - used,
        "exclusion_reason": ", ".join(reasons),
    }


def load_frame_and_groups(
    cfg: PipelineConfig, sample: SampleSpec | None
) -> tuple[pd.DataFrame, str, str, dict[str, np.ndarray], dict[str, int | str]]:
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
    attrition = _attrition(df, source_group_column, group_values, cfg.dataset.target)
    return df, group_column, source_group_column, groups, attrition


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
    figures_dir: Path,
    attrition: dict[str, int | str] | None = None,
) -> AnalysisStep:
    summary = describe(
        df, group_column, source_group_column, group_values, target,
        source_path, sample_name or "canonical", "diagnostic",
    )
    if attrition is None:
        attrition = _attrition(df, source_group_column, group_values, target)
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "describe.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    describe_figure = load_viz_config().describe_figure
    plot_describe_figures(
        df, source_group_column, group_column, group_values, target, summary,
        figures_dir / describe_figure,
    )
    thresholds = _thresholds()
    flagged = summary.imbalance_ratio > thresholds.imbalance_ratio_threshold or bool(summary.absent_groups)
    if flagged:
        parts = []
        if summary.absent_groups:
            parts.append(f"groups {', '.join(summary.absent_groups)} have no observations")
        if summary.imbalance_ratio > thresholds.imbalance_ratio_threshold:
            parts.append(f"group sizes are imbalanced (imbalance ratio {humanize_float(summary.imbalance_ratio)})")
        ramification = "; ".join(parts) + "."
    else:
        ramification = "group sizes are adequate."
    return AnalysisStep(
        analysis=analysis.name,
        step_id="describe_groups",
        order=order,
        question=question,
        status=StepStatus.NOTE if flagged else StepStatus.COMPLETED,
        method="describe",
        source=source,
        sample_name=sample_name,
        evidence_refs=[
            "describe.json",
            f"figures/{describe_figure}",
        ],
        figures=[
            FigureRef(path=f"figures/{describe_figure}", caption=BOXPLOT_CAPTION),
        ],
        result_summary={
            "imbalance_ratio": summary.imbalance_ratio,
            "absent_groups": len(summary.absent_groups),
            "n_total": attrition["n_total"],
            "n_used": attrition["n_used"],
            "n_excluded": attrition["n_excluded"],
            "exclusion_reason": attrition["exclusion_reason"],
        },
        ramification=ramification,
        decision_required=False,
        performed_at=now_iso(),
    )


def _plot_qq_joint(
    groups: dict[str, np.ndarray], out_path: Path, max_groups: int | None = None
) -> int:
    if max_groups is None:
        max_groups = load_walkthrough_config().max_qq_groups
    viz_cfg = load_viz_config()
    names = list(groups)[:max_groups]
    n = len(names)
    n_cols = max(1, int(np.ceil(np.sqrt(n))))
    n_rows = max(1, int(np.ceil(n / n_cols)))
    colors = viz.palette_colors(n)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * viz_cfg.fig_size_per_subplot, n_rows * viz_cfg.fig_size_per_subplot),
        squeeze=False,
        layout="constrained",
    )
    ax_flat = axes.ravel()
    skipped: list[str] = []
    plotted = 0
    total = 0
    for name in names:
        vals = np.asarray(groups[name], dtype=float)
        std = vals.std()
        if std == 0.0 or not np.isfinite(std):
            skipped.append(name)
            continue
        total += len(vals)
        ax = ax_flat[plotted]
        z = (vals - vals.mean()) / std
        (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm", fit=True)
        ax.scatter(
            osm,
            osr,
            s=viz.QQ_SCATTER_SIZE,
            alpha=viz.QQ_SCATTER_ALPHA,
            edgecolor=viz.QQ_SCATTER_EDGE_COLOR,
            color=colors[plotted],
        )
        xs = np.array([osm.min(), osm.max()])
        ax.plot(
            xs,
            slope * xs + intercept,
            color=viz.QQ_REF_LINE_COLOR,
            linestyle=viz.QQ_REF_LINE_STYLE,
            linewidth=viz.QQ_REF_LINE_WIDTH,
        )
        ax.set_xlabel(viz.QQ_XLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_ylabel(viz.QQ_YLABEL, fontsize=viz.LABEL_FONTSIZE)
        ax.set_title(name, fontsize=viz.TITLE_FONTSIZE)
        ax.grid(True, alpha=viz.GRID_ALPHA)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
        plotted += 1
    for ax in ax_flat[plotted:]:
        ax.set_visible(False)
    if skipped:
        logger.warning(
            "Q-Q plot skipped %d zero-variance group(s): %s",
            len(skipped),
            ", ".join(skipped),
        )
    fig.suptitle(
        f"Per-group Q-Q plots (per-group standardization) — n = {total:,}",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
    fig.savefig(out_path, dpi=viz_cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return plotted


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
    thresholds = _thresholds()
    max_qq_groups = thresholds.max_qq_groups
    normality_figure = load_viz_config().normality_figure
    _plot_qq_joint(groups, figures_dir / normality_figure, max_qq_groups)
    figure_path = f"figures/{normality_figure}"
    figure_refs = [FigureRef(path=figure_path, caption=QQ_CAPTION)]
    evidence = NormalityEvidence(
        groups={g: NormalityGroupStat(**s) for g, s in result.items()},
        figure=figure_path,
    )
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "normality.json").write_text(
        evidence.model_dump_json(indent=2), encoding="utf-8"
    )
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
    result_summary: dict[str, object] = {
        g: {"skew": s["skew"], "kurtosis": s["kurtosis"], "shapiro_p": s["shapiro_p"]}
        for g, s in result.items()
    }
    result_summary["standardization"] = "per-group z-score"
    if len(groups) > max_qq_groups:
        result_summary["truncation"] = (
            f"plotted first {max_qq_groups} of {len(groups)} groups"
        )
    return AnalysisStep(
        analysis=analysis.name,
        step_id="normality",
        order=order,
        question=question,
        status=StepStatus.NOTE if flagged else StepStatus.COMPLETED,
        method="check_normality",
        source=source,
        sample_name=sample_name,
        evidence_refs=["normality.json", figure_path],
        figures=figure_refs,
        result_summary=result_summary,
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
    flagged = result["p_value"] < thresholds.significance_alpha
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
    thresholds = _thresholds()
    alpha = thresholds.significance_alpha
    if method == "anova":
        plan = run_anova(groups, alpha=alpha)
    elif method == "welch":
        plan = run_welch(groups, alpha=alpha)
    elif method == "kruskal":
        plan = run_kruskal(groups, alpha=alpha)
    else:
        raise ValueError(f"unknown omnibus method '{method}'")
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "omnibus.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    p_value = plan.statistics["p_value"]
    p_text = humanize_pvalue(p_value)
    if plan.passed:
        ramification = f"reject H0: at least one group mean differs (p={p_text})"
    else:
        ramification = f"fail to reject H0: no mean difference (p={p_text})"
    if method in ("anova", "welch"):
        eta = plan.effect_sizes["eta_squared"]
        omega = plan.effect_sizes["omega_squared"]
        ramification += (
            f" eta² = {humanize_float(eta)} (proportion of outcome variance explained by"
            " group membership; can be inflated under extreme imbalance);"
            f" omega² = {humanize_float(omega)} (corrects for small-sample bias; the more"
            " conservative estimate)."
        )
        imbalance_ratio = plan.threshold_context.get("imbalance_ratio")
        if imbalance_ratio is not None and float(imbalance_ratio) > thresholds.imbalance_ratio_threshold:
            ramification += (
                " eta² and omega² diverge here because group sizes are extremely imbalanced;"
                " report omega²."
            )
    elif method == "kruskal":
        eps = plan.effect_sizes["epsilon_squared"]
        ramification += (
            f" rank-based ε² = {humanize_float(eps)}"
            " (proportion of variance in ranks explained by group membership)."
        )
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
        result_summary["epsilon_squared"] = plan.effect_sizes["epsilon_squared"]
    return AnalysisStep(
        analysis=analysis.name,
        step_id="omnibus",
        order=order,
        question=question,
        status=StepStatus.NOTE if plan.warnings else StepStatus.COMPLETED,
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
            "significant_pair_details": [
                {
                    "a": p.a,
                    "b": p.b,
                    "p_value": p.p_value,
                    "cohens_d": p.cohens_d,
                    "hedges_g": p.hedges_g,
                    "effect_size_note": p.effect_size_note,
                }
                for p in pairs
                if p.p_value < thresholds.significance_alpha
            ],
        },
        ramification=(
            f"Games-Howell found {significant_pairs} significant pairwise "
            f"difference(s) at alpha={thresholds.significance_alpha}."
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
    p_text = humanize_pvalue(p_value)
    if method in ("anova", "welch"):
        effect_size = (
            f"eta²={humanize_float(float(summary['eta_squared']))}, "
            f"omega²={humanize_float(float(summary['omega_squared']))}"
        )
    else:
        effect_size = f"rank-based ε² = {humanize_float(float(summary['epsilon_squared']))}"
    significant_pairs = (
        int(posthoc_step.result_summary.get("significant_pairs", 0))
        if posthoc_step is not None
        else 0
    )
    if passed:
        verdict = (
            f"group means differ ({method} p={p_text}), with {significant_pairs} "
            "significant pairwise difference(s)."
        )
    else:
        verdict = f"no significant difference across groups ({method} p={p_text})."
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
    result_summary: dict[str, object] = {
        "verdict": verdict,
        "principal_method": method,
        "p_value": p_value,
        "significant_pairs": significant_pairs,
    }
    if method in ("anova", "welch") and "eta_squared" in summary and "omega_squared" in summary:
        result_summary["eta_squared"] = summary["eta_squared"]
        result_summary["omega_squared"] = summary["omega_squared"]
    elif "epsilon_squared" in summary:
        result_summary["epsilon_squared"] = float(summary["epsilon_squared"])
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
        result_summary=result_summary,
        ramification=verdict,
        decision_required=False,
        performed_at=now_iso(),
    )
