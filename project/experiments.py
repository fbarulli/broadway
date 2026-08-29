"""Combined project experiment scripts.

Four loose root scripts merged into one module behind a subcommand dispatcher:

- ``ols``        — capped-stratified OLS baseline (Manhattan) vs incremental
                   borough pooling, saving residual Q-Q plots per step
                   (formerly experiment_ols.py).
- ``diagnostics`` — distribution-diagnostics redesign: zscore/ratio/bars
                   renderings of the per-feature diagnostics surface
                   (formerly experiment_diagnostics.py).
- ``qq_legend``  — Q-Q legend placement experiment (right side) across all
                   plot surfaces (formerly experiment_qq_legend.py).
- ``verify``     — lightweight verification of the project experiment tree and the
                   experiment config YAMLs (formerly verify_experiments.py).

Usage:

    uv run python project/experiments.py <command>
"""

import argparse
import ast
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy import stats

from broadway import viz
from broadway.config.viz import load_viz_config
from broadway.discover import qq
from broadway.evaluate.metrics import binary_metrics, compute_metrics
from broadway.lineage.sample import load_sample
from broadway.stats import diagnostics, regression
from broadway.stats.describe import describe, plot_describe_figures
from broadway.stats.robust import winsorize
from broadway.training.optuna_worker import compose_db_url
from broadway.utils import require_keys
from project.data import DATA_PATH, LOOKUP_PATH
from project.paths import load_project_paths
from project.working import MIN_TARGET_VALUE, TARGET_COL

# Shared constants (identical across the merged scripts).
PATHS = load_project_paths()
REPO = PATHS.root.parent
TRAINING = DATA_PATH
EXCLUDE = ["pickup_location_id", "dropoff_location_id"]
TARGET = "trip_duration_minutes"

# ols (formerly experiment_ols.py).
OLS_OUT = PATHS.results / "ols"
LOOKUP = LOOKUP_PATH
OLS_KEEP = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
DISTANCE = "trip_distance"
BOROUGH = "pickup_borough"
SEED = 42

# diagnostics (formerly experiment_diagnostics.py).
DIAG_OUT = PATHS.results / "diagnostics"
METRICS = ["skew", "kurtosis", "zero_rate", "max/p99"]

# qq_legend (formerly experiment_qq_legend.py).
LEGEND_OUT = PATHS.results / "qq_legend"
SAMPLE = (REPO / load_sample("taxi_diagnostic").path).resolve()
GROUP_COLUMN = "Borough"
SOURCE_GROUP = "pickup_borough"
GROUP_VALUES = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

# verify (formerly verify_experiments.py).
EXPERIMENTS = PATHS.experiments
UNIVARIATE = EXPERIMENTS / "univariate" / "fare_amount_trip_distance"
MULTIVARIATE = EXPERIMENTS / "multivariate"
MLFLOW = EXPERIMENTS / "mlflow"
# The k8s configmap carries infra only (dataset/databases/mlflow); the HPO
# spec (search spaces, budgets) lives in the project experiment config.
K8S_CONFIG_KEYS = ["dataset", "databases", "mlflow"]


def build_capped_sample(cap: int | None = None) -> pd.DataFrame:
    zones = pl.read_csv(LOOKUP).select([pl.col("LocationID"), pl.col("Borough")]).lazy()
    full = (
        pl.scan_parquet(TRAINING)
        .select([pl.col("pickup_location_id"), pl.col(DISTANCE), pl.col(TARGET)])
        .join(zones, left_on="pickup_location_id", right_on="LocationID", how="left")
        .filter(pl.col("Borough").is_in(OLS_KEEP))
        .collect()
    )
    if cap is None:
        cap = int(full.group_by("Borough").len().select(pl.col("len").min()).item())
    parts = []
    for b in OLS_KEEP:
        g = full.filter(pl.col("Borough") == b)
        parts.append(g.sample(n=min(cap, g.height), seed=SEED))
    return (
        pl.concat(parts)
        .select([pl.col("Borough").alias(BOROUGH), pl.col(DISTANCE), pl.col(TARGET)])
        .to_pandas()
    )


def fit_and_plot(df: pd.DataFrame, out_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = regression.fit_ols(df, f"{TARGET} ~ {DISTANCE}")
    diagnostics.plot_residuals(model, str(out_dir / "residuals.png"))
    result = regression.bp_jb(model)
    dw = diagnostics.durbin_watson(model.resid)
    row = {
        "step": label,
        "n": len(df),
        "r2": round(float(model.rsquared), 4),
        "jb_skew": round(result["skew"], 3),
        "jb_kurtosis": round(result["kurtosis"], 3),
        "jb_p": result["jb_pval"],
        "dw": round(dw, 3),
        "boroughs": sorted(df[BOROUGH].unique().tolist()),
    }
    print(row)
    return row


def main_ols() -> None:
    sample = build_capped_sample()
    print(f"capped stratified sample: {len(sample)} rows")
    print(sample.groupby(BOROUGH).size().to_string())
    steps = [
        ("manhattan", ["Manhattan"]),
        ("plus_queens", ["Manhattan", "Queens"]),
        ("plus_brooklyn", ["Manhattan", "Queens", "Brooklyn"]),
        ("plus_bronx", ["Manhattan", "Queens", "Brooklyn", "Bronx"]),
    ]
    for label, boroughs in steps:
        df = sample[sample[BOROUGH].isin(boroughs)].copy()
        fit_and_plot(df, OLS_OUT / label, label)
    print("done")


def right_side_legend(fig, zones, any_shelf, markers=None) -> None:
    handles = qq.build_qq_legend_handles(zones, any_shelf, markers)
    if not handles:
        return
    fig.set_layout_engine(None)
    fig_w = fig.get_size_inches()[0]
    frac = min(1.7 / fig_w, 0.4)
    fig.subplots_adjust(right=1.0 - frac)
    band = fig.add_axes([1.0 - frac, 0.0, frac, 1.0])
    band.axis("off")
    band.legend(handles=handles, loc="center", ncol=1,
                frameon=False, fontsize=viz.TICK_FONTSIZE)


def diag_rows(overview):
    rows = []
    for f in overview.features:
        if f.status not in ("plotted", "discrete"):
            continue
        if f.skew is None or f.kurtosis is None or f.zero_rate is None:
            continue
        max_p99 = (
            f.max / f.p99
            if f.p99 is not None and f.max is not None and f.p99 > 0
            else None
        )
        rows.append((f.feature, f.skew, f.kurtosis, f.zero_rate, max_p99))
    return rows


def thresholds():
    t = load_viz_config().diagnostics.thresholds
    return [t.skew, t.kurtosis, t.zero_rate, t.max_p99_ratio]


def raw_matrix(rows):
    m = []
    for name, sk, ku, zr, mp in rows:
        m.append([sk, ku, zr, mp if mp is not None else np.nan])
    return np.array(m, dtype=float)


def plot_ratio_heatmap(rows, out_path, thr, dpi=100) -> None:
    names = [r[0] for r in rows]
    raw = raw_matrix(rows)
    ratio = np.clip(raw / np.array(thr), 0.0, 2.0)
    fig, ax = plt.subplots(
        figsize=(6.0, max(2.0, 0.35 * len(rows))), layout="constrained",
    )
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=2.0)
    ax.imshow(ratio, aspect="auto", cmap="YlOrRd", norm=norm)
    ax.set_xticks(range(raw.shape[1]))
    ax.set_xticklabels(METRICS, fontsize=viz.TICK_FONTSIZE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=viz.TICK_FONTSIZE)
    for i in range(len(rows)):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            if np.isnan(val):
                continue
            text_color = "black" if ratio[i, j] < 1.0 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=viz.TICK_FONTSIZE, color=text_color)
    fig.colorbar(ax.images[0], ax=ax, label="value / threshold (1.0 = flagging boundary)")
    viz.despine(ax)
    fig.suptitle(
        "Per-feature diagnostics — value / threshold (capped at 2.0)",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_bars(rows, out_path, thr, dpi=100) -> None:
    n = len(rows)
    fig, axes = plt.subplots(
        4, 1, figsize=(8.0, max(6.0, 0.6 * n + 2.0)), layout="constrained",
    )
    for ax, (label, idx, t) in zip(axes, [
        ("skew", 1, thr[0]),
        ("kurtosis", 2, thr[1]),
        ("zero_rate", 3, thr[2]),
        ("max/p99", 4, thr[3]),
    ]):
        valid = [(r[0], r[idx]) for r in rows if r[idx] is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        names = [v[0] for v in valid]
        vals = [v[1] for v in valid]
        colors = ["#d62728" if v > t else "#bbbbbb" for v in vals]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, height=0.7)
        ax.axvline(t, color="#333333", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=viz.TICK_FONTSIZE)
        ax.invert_yaxis()
        ax.set_xlabel(f"{label} (threshold {t})", fontsize=viz.LABEL_FONTSIZE)
        ax.tick_params(labelsize=viz.TICK_FONTSIZE)
        viz.despine(ax)
    fig.suptitle(
        "Per-feature diagnostics — sorted by metric, red = exceeds threshold",
        fontsize=viz.SUPTITLE_FONTSIZE,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main_diagnostics() -> None:
    viz_cfg = load_viz_config()
    thr = thresholds()
    training = pd.read_parquet(TRAINING)

    qq.attach_qq_legend = right_side_legend
    overview = qq.plot_numeric_qq(
        training, DIAG_OUT / "zscore", DIAG_OUT / "zscore" / "qq_overview.json",
        source_path=str(TRAINING), exclude=EXCLUDE,
    )
    rows = diag_rows(overview)

    (DIAG_OUT / "ratio").mkdir(parents=True, exist_ok=True)
    (DIAG_OUT / "bars").mkdir(parents=True, exist_ok=True)

    plot_ratio_heatmap(rows, DIAG_OUT / "ratio" / "diagnostics_ratio.png", thr, dpi=viz_cfg.dpi)
    plot_bars(rows, DIAG_OUT / "bars" / "diagnostics_bars.png", thr, dpi=viz_cfg.dpi)
    print("done")


def patched_legend_factory(mode):
    def patched(fig, zones, any_shelf, markers=None):
        handles = qq.build_qq_legend_handles(zones, any_shelf, markers)
        if not handles:
            return
        fig.set_layout_engine(None)
        fig_w = fig.get_size_inches()[0]
        legend_w = 1.7
        frac = min(legend_w / fig_w, 0.4)
        fig.subplots_adjust(right=1.0 - frac)
        band = fig.add_axes([1.0 - frac, 0.0, frac, 1.0])
        band.axis("off")
        band.legend(handles=handles, loc="center", ncol=1,
                    frameon=False, fontsize=viz.TICK_FONTSIZE)
    return patched


def render_features(df, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    qq.plot_numeric_qq(
        df, out_dir, out_dir / "qq_overview.json",
        source_path=str(TRAINING), exclude=EXCLUDE,
    )


def render_groups(sample_df, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg = load_viz_config()
    groups = {
        g: sample_df[sample_df[SOURCE_GROUP] == g][TARGET].dropna().to_numpy()
        for g in GROUP_VALUES
    }
    pooled = np.concatenate(list(groups.values()))
    show_log = bool(
        pooled.size > 1
        and pooled.min() > 0
        and float(stats.skew(pooled)) > viz_cfg.diagnostics.thresholds.skew
    )
    qq._plot_qq_joint(
        groups, out_dir / viz_cfg.normality_figure, None,
        show_log=show_log, markers=viz_cfg.qq_markers,
    )


def render_describe(sample_df, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = describe(
        sample_df, GROUP_COLUMN, SOURCE_GROUP, GROUP_VALUES, TARGET,
        str(SAMPLE), "canonical", "diagnostic",
    )
    plot_describe_figures(
        sample_df, SOURCE_GROUP, GROUP_COLUMN, GROUP_VALUES, TARGET, summary,
        out_dir / load_viz_config().describe_figure,
    )


def main_qq_legend() -> None:
    training = pd.read_parquet(TRAINING)
    sample = pd.read_parquet(SAMPLE)
    for mode in ("right",):
        qq.attach_qq_legend = patched_legend_factory(mode)
        out = LEGEND_OUT / mode
        print(f"rendering {mode} -> {out}")
        render_features(training, out)
        render_groups(sample, out)
        render_describe(sample, out)
    print("done")


def load_module(name: str, path: Path):
    """Load a module from a file under a unique name (no shadowing)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_experiments() -> list[str]:
    """Syntax-compile every experiment script; return problem files."""
    bad = []
    for py in sorted(EXPERIMENTS.rglob("*.py")):
        try:
            ast.parse(py.read_text(), filename=str(py))
        except SyntaxError:
            bad.append(str(py))
    return bad


def validate_configs() -> list[str]:
    """Validate the YAML configs against require_keys; return problems."""
    problems = []
    for name, keys in (
        ("multivariate.yaml", ["target", "categorical", "sample", "baseline"]),
        ("working.yaml", ["parquet", "columns", "min_target_value",
                          "max_duration_minutes", "time_buckets"]),
        ("mlflow.yaml", ["sample_size", "test_fraction", "seed",
                         "continuous_features", "categorical_features"]),
    ):
        cfg = yaml.safe_load((PATHS.experiment_configs / name).read_text())
        try:
            require_keys(cfg, keys, name)
        except ValueError as exc:
            problems.append(str(exc))
    k8s_cfg = yaml.safe_load((REPO / "k8s" / "optuna" / "configmap.yaml")
                             .read_text())["data"]["config.yaml"]
    try:
        require_keys(yaml.safe_load(k8s_cfg), K8S_CONFIG_KEYS, "k8s config.yaml")
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def check_univariate() -> list[str]:
    """Load the univariate loader and check structural invariants."""
    problems = []
    common = load_module("_uni_verify_common", UNIVARIATE / "_common.py")
    sample = common.load_metered()
    if sample.empty:
        problems.append("univariate: load_metered returned no rows")
    if not (sample[TARGET_COL] > MIN_TARGET_VALUE).all():
        problems.append("univariate: min_target_value filter not honored")
    if not (sample["duration_minutes"] < common.MAX_DURATION_MINUTES).all():
        problems.append("univariate: max-duration filter not honored")
    if "trip_distance" not in sample.columns:
        problems.append("univariate: missing trip_distance column")
    return problems


def check_multivariate() -> list[str]:
    """Load the multivariate setup + config; check the sample + dummies.

    The borough join needs a zones lookup. Rather than require the real
    taxi data file (absent in CI), generate a minimal zones CSV from the
    sample's own location IDs and point the setup at it.
    """
    problems = []
    setup = load_module("_mv_verify_setup", MULTIVARIATE / "_setup.py")
    cfg = setup.load_config()
    sample = setup.load_metered()
    zones_dir = Path(tempfile.mkdtemp(prefix="broadway_verify_"))
    location_ids = sorted(sample[cfg["borough"]["pickup"]["join_on"]].unique())
    keep = cfg["sample"]["pickup_borough"]
    # Label half the locations as the config's keep borough so the
    # Manhattan filter selects a substantial subset; rest are "Other".
    half = max(1, len(location_ids) // 2)
    zones = pd.DataFrame({
        setup.ZONE_ID_COL: location_ids,
        setup.ZONE_BOROUGH_COL: [keep] * half + ["Other"] * (len(location_ids) - half),
    })
    zones_path = zones_dir / "zones.csv"
    zones.to_csv(zones_path, index=False)
    setup.LOOKUP_PATH = zones_path

    manhattan = setup.load_manhattan_sample(cfg)
    pickup_col = cfg["borough"]["pickup"]["column"]
    keep = cfg["sample"]["pickup_borough"]
    if manhattan.empty:
        problems.append("multivariate: manhattan sample empty")
    if not (manhattan[pickup_col] == keep).all():
        problems.append("multivariate: pickup filter not honored")
    dummies = setup.build_borough_dummies(manhattan.head(50), cfg)
    if dummies.shape[0] != min(50, len(manhattan)):
        problems.append("multivariate: borough dummies row mismatch")
    return problems


def check_mlflow() -> list[str]:
    """Import the mlflow experiment module; spot-check metrics + url helper."""
    problems = []
    load_module("_mlflow_verify_common", MLFLOW / "_common.py")
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    if set(metrics) != {"mae", "rmse", "r2", "mape", "max_error",
                        "median_ae", "explained_var"}:
        problems.append("mlflow: compute_metrics key set mismatch")
    binm = binary_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], threshold=1.5)
    if set(binm) != {"roc_auc", "pr_auc"}:
        problems.append("mlflow: binary_metrics key set mismatch")
    url = compose_db_url("postgresql+psycopg2", "u", "p", "h", "5432", "d")
    if url != "postgresql+psycopg2://u:p@h:5432/d":
        problems.append("mlflow: compose_db_url mismatch")
    return problems


def check_robust() -> list[str]:
    """Spot-check the promoted robust helper on the real sample."""
    problems = []
    common = load_module("_uni_verify_robust", UNIVARIATE / "_common.py")
    sample = common.load_metered()
    clipped = winsorize(sample, ["fare_amount", "trip_distance"], 0.995)
    for col in ("fare_amount", "trip_distance"):
        cap = sample[col].quantile(0.995)
        if (clipped[col] > cap).any() or clipped[col].isna().any():
            problems.append(f"robust: winsorize invariant broken for {col}")
    return problems


def main_verify() -> int:
    checks = {
        "compile experiments": compile_experiments,
        "validate configs": validate_configs,
        "univariate loader": check_univariate,
        "multivariate setup": check_multivariate,
        "mlflow + metrics": check_mlflow,
        "robust helpers": check_robust,
    }
    failed = False
    for label, fn in checks.items():
        problems = fn()
        if problems:
            failed = True
            print(f"FAIL  {label}:")
            for p in problems:
                print(f"      - {p}")
        else:
            print(f"PASS  {label}")
    if failed:
        print("\nverification FAILED")
        return 1
    print("\nverification OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="experiments", description="Combined root experiment scripts (ols, diagnostics, qq_legend, verify).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ols", help="capped-stratified OLS baseline vs incremental borough pooling")
    p.set_defaults(func=main_ols)
    p = sub.add_parser("diagnostics", help="distribution-diagnostics redesign (zscore/ratio/bars)")
    p.set_defaults(func=main_diagnostics)
    p = sub.add_parser("qq_legend", help="Q-Q legend placement experiment (right side)")
    p.set_defaults(func=main_qq_legend)
    p = sub.add_parser("verify", help="lightweight verification of the project experiment tree + configs")
    p.set_defaults(func=main_verify)
    args = parser.parse_args()
    result = args.func()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
