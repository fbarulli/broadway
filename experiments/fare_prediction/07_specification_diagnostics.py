"""07: specification diagnostics — Ramsey RESET, Pearson-vs-Spearman discrepancies, MI-vs-R² audit; now audits the SAFE (pre-trip) feature set — post-trip columns (distance/duration/speed) are leakage and are excluded."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from _common import RESULTS, SAFE_NUMERIC_FEATURES
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import linear_reset

PREPARED_TRAIN = RESULTS / "prepared" / "train.parquet"
TARGET = "fare_amount"
RESET_X_COLS = ["pickup_hour", "route_id_encoded", "pu_rush_interaction"]
# The safe-feature nonlinearity check: route_id_encoded is the strongest safe
# association; a quadratic term tests whether its curve needs capturing.
RESET_VARIANTS = (
    ("linear (safe)", RESET_X_COLS),
    ("quadratic route_id", ["pickup_hour", "route_id_encoded", "route_id_sq", "pu_rush_interaction"]),
)
AUDIT_COLS = list(SAFE_NUMERIC_FEATURES)
MI_NEIGHBORS = 5
MI_RANDOM_STATE = 42
DELTA_THRESHOLD = 0.05
R2_FLOOR = 0.01
N_FITTED_BINS = 10
SCATTER_SAMPLE = 25_000

DESCRIBE_CSV = RESULTS / "07_specification_diagnostics_describe.csv"
PNG_MAIN = RESULTS / "07_specification_diagnostics.png"
PNG_RESET = RESULTS / "07_specification_diagnostics_reset.png"
SUMMARY_MD = RESULTS / "07_specification_diagnostics.md"


def _corr(stat: object) -> float:
    """Extract a scipy correlation coefficient (``.statistic`` on 1.11+, else ``[0]``)."""
    return float(stat.statistic) if hasattr(stat, "statistic") else float(stat[0])


def fit_linear_baseline(df: pd.DataFrame) -> RegressionResultsWrapper:
    """Fit the plain-OLS RESET baseline: target ~ RESET_X_COLS + constant."""
    return sm.OLS(df[TARGET], sm.add_constant(df[RESET_X_COLS])).fit()


def run_reset(model: RegressionResultsWrapper) -> dict[int, tuple[float, float]]:
    """Run ``linear_reset`` (yhat² and yhat³ terms), print F/p/verdict per power."""
    results: dict[int, tuple[float, float]] = {}
    for power in (2, 3):
        test = linear_reset(model, power=power, use_f=True)
        f_value, p_value = float(test.statistic), float(test.pvalue)
        results[power] = (f_value, p_value)
        verdict = (
            "specification error (p<0.05)"
            if p_value < 0.05
            else "no evidence of misspecification"
        )
        print(f"RESET power={power}: F={f_value:,.1f}, p={p_value:.3e} -> {verdict}")
    return results


def _variant_frame(df: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    """X frame with the transforms a RESET variant needs."""
    base_cols = [c for c in x_cols if c not in ("route_id_sq",)]
    out = df[base_cols].copy()
    if "route_id_sq" in x_cols:
        out["route_id_sq"] = df["route_id_encoded"] ** 2
    return out


def run_reset_variants(df: pd.DataFrame) -> pd.DataFrame:
    """RESET (power=2) across the linear and quadratic-route_id specs.

    route_id_encoded is the strongest safe association; the quadratic term
    tests whether its curve needs capturing (the safe-set analogue of the
    earlier monotonic-trap check on duration).
    """
    rows = []
    for label, x_cols in RESET_VARIANTS:
        X = _variant_frame(df, x_cols)
        model = sm.OLS(df[TARGET], sm.add_constant(X)).fit()
        test = linear_reset(model, power=2, use_f=True)
        rows.append({
            "model": label,
            "r2": float(model.rsquared),
            "reset_f": float(test.statistic),
            "reset_p": float(test.pvalue),
        })
    variants = pd.DataFrame(rows).set_index("model")
    print("\nRESET variants (safe features — route_id_encoded curvature):")
    print(variants.round(4).to_string())
    return variants


def correlation_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Per audit feature: Pearson r, Spearman r, and |pearson − spearman|."""
    y = df[TARGET]
    rows = []
    for col in AUDIT_COLS:
        pearson = _corr(pearsonr(df[col], y))
        spearman = _corr(spearmanr(df[col], y))
        rows.append(
            {"pearson": pearson, "spearman": spearman, "delta": abs(pearson - spearman)}
        )
    return pd.DataFrame(rows, index=pd.Index(AUDIT_COLS, name="feature"))


def mi_r2_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Per audit feature: mutual information (nats), R² (pearson²), and MI/R² ratio."""
    mi = mutual_info_regression(
        df[AUDIT_COLS], df[TARGET], random_state=MI_RANDOM_STATE, n_neighbors=MI_NEIGHBORS
    )
    rows = []
    for col, value in zip(AUDIT_COLS, mi):
        r2 = _corr(pearsonr(df[col], df[TARGET])) ** 2
        rows.append(
            {
                "r2": r2,
                "mi": float(value),
                "mi_r2_ratio": float(value) / max(r2, R2_FLOOR),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(AUDIT_COLS, name="feature"))


def _bars(ax: plt.Axes, frame: pd.DataFrame, cols: list[str], title: str) -> None:
    """Grouped barplot of ``cols`` across the (caller-sorted) feature rows."""
    melted = frame.reset_index().melt(
        id_vars=frame.index.name or "feature", value_vars=cols,
        var_name="metric", value_name="value",
    )
    sns.barplot(
        data=melted, x="feature", y="value", hue="metric", ax=ax,
        palette=["#4c72b0", "#dd8452"],
    )
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def _diag_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    label_top: int = 0,
) -> None:
    """Scatter of two metrics with a y=x reference line and optional top-N labels."""
    ax.scatter(frame[x_col], frame[y_col], s=30, color="#4c72b0", alpha=0.85)
    lo = min(frame[x_col].min(), frame[y_col].min())
    hi = max(frame[x_col].max(), frame[y_col].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#7f7f7f", lw=1.0)
    if label_top:
        for name, row in frame.nlargest(label_top, "mi_r2_ratio").iterrows():
            ax.annotate(
                name, (row[x_col], row[y_col]), fontsize=8, color="#d62728",
                xytext=(4, 4), textcoords="offset points",
            )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_main(evidence: pd.DataFrame, out_path: Path) -> None:
    """4 panels: association bars (Pearson/Spearman, MI/R²) and two y=x scatter audits."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _bars(axes[0, 0], evidence.sort_values("delta", ascending=False),
          ["pearson", "spearman"],
          "Pearson vs Spearman with fare_amount (sorted by |Δ|)")
    _bars(axes[0, 1], evidence.sort_values("mi_r2_ratio", ascending=False),
          ["mi", "r2"],
          "MI (nats) vs R² (pearson²), sorted by MI/R² ratio")
    _diag_scatter(axes[1, 0], evidence, "pearson", "spearman",
                  "Pearson vs Spearman — off-diagonal = nonlinear/outlier-driven")
    _diag_scatter(axes[1, 1], evidence, "r2", "mi",
                  "MI vs R² — above y=x = nonlinear association the linear model misses",
                  label_top=3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _variance_by_bin(ax: plt.Axes, model: RegressionResultsWrapper) -> None:
    """Bar of residual variance within each quantile bin of fitted values (ascending)."""
    bins = pd.qcut(model.fittedvalues, N_FITTED_BINS, duplicates="drop")
    var_by_bin = (model.resid**2).groupby(bins, observed=True).mean()
    labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in var_by_bin.index]
    ax.bar(range(len(var_by_bin)), var_by_bin.values, color="#4c72b0")
    ax.set_xticks(range(len(var_by_bin)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("residual variance by fitted bin (linear baseline)")
    ax.set_xlabel("fitted fare_amount bin")
    ax.set_ylabel("residual variance")
    ax.grid(True, alpha=0.3, axis="y")


def _residuals_fitted(
    ax: plt.Axes, model: RegressionResultsWrapper, reset: dict[int, tuple[float, float]]
) -> None:
    """Subsampled residuals-vs-fitted scatter with binned means and the RESET annotation."""
    rng = np.random.default_rng(MI_RANDOM_STATE)
    idx = rng.choice(len(model.resid), SCATTER_SAMPLE, replace=False)
    ax.scatter(np.asarray(model.fittedvalues)[idx], np.asarray(model.resid)[idx],
               s=3, alpha=0.25, color="#4c72b0")
    bins = pd.qcut(model.fittedvalues, N_FITTED_BINS, duplicates="drop")
    means = model.resid.groupby(bins, observed=True).mean()
    ax.plot([iv.mid for iv in means.index], means.values, "o-",
            color="#d62728", ms=4, lw=1.2)
    ax.axhline(0.0, color="#7f7f7f", lw=0.8, linestyle="--")
    text = "\n".join(
        f"RESET power={power}: F={f:,.0f}, p={p:.2e}" for power, (f, p) in reset.items()
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=7, va="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#ffffff", "ec": "#cccccc"})
    ax.set_title("residuals vs fitted (yhat² curve = the RESET signal)")
    ax.set_xlabel("fitted fare_amount")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.3)


def _route_curve(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Fare vs route_id_encoded: linear / log / quadratic fits over the same curve."""
    sample = df.sample(n=min(SCATTER_SAMPLE, len(df)), random_state=MI_RANDOM_STATE)
    d = sample["route_id_encoded"]
    y = sample[TARGET]
    ax.scatter(d, y, s=3, alpha=0.12, color="#4c72b0")
    grid = np.linspace(d.quantile(0.01), d.quantile(0.99), 100)
    ax.plot(grid, np.polyval(np.polyfit(d, y, 1), grid), "-",
            color="#dd8452", lw=1.6, label="linear")
    ax.plot(grid, np.polyval(np.polyfit(np.log1p(d), y, 1), np.log1p(grid)), "--",
            color="#2ca02c", lw=1.6, label="log(route_id_encoded)")
    ax.plot(grid, np.polyval(np.polyfit(d, y, 2), grid), ":",
            color="#d62728", lw=1.6, label="quadratic")
    ax.set_xlabel("route_id_encoded")
    ax.set_ylabel("fare_amount")
    ax.set_title("fare vs route_id_encoded — linear vs log vs quadratic")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_reset(
    model: RegressionResultsWrapper,
    reset: dict[int, tuple[float, float]],
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """3 panels: residual-variance-by-bin bars, residuals-vs-fitted, route curve."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    _variance_by_bin(axes[0], model)
    _residuals_fitted(axes[1], model, reset)
    _route_curve(axes[2], df)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a feature-indexed frame as an aligned Markdown pipe table (03/04 style)."""
    rows = []
    for name, row in frame.iterrows():
        rows.append([str(name), *[f"{value:.4f}" for value in row]])
    header = ["feature", *frame.columns]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def fmt(cells: list[str], aligns: list[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(w) if a == "r" else cell.ljust(w)
            for cell, w, a in zip(cells, widths, aligns)
        ) + " |"

    aligns = ["l"] + ["r"] * (len(header) - 1)
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join([fmt(header, aligns), sep] + [fmt(row, aligns) for row in rows])
    return body


def _summary_md(
    n_rows: int,
    evidence: pd.DataFrame,
    reset: dict[int, tuple[float, float]],
    variants: pd.DataFrame,
) -> str:
    """Render the specification-diagnostics Markdown summary."""
    f2, p2 = reset[2]
    f3, p3 = reset[3]
    verdict2 = "specification error (p<0.05)" if p2 < 0.05 else "no evidence of misspecification"
    verdict3 = "specification error (p<0.05)" if p3 < 0.05 else "no evidence of misspecification"
    delta_flagged = evidence[evidence["delta"] >= DELTA_THRESHOLD].index.tolist()
    top_delta = evidence.nlargest(3, "delta")[["pearson", "spearman", "delta"]]
    top_ratio = evidence.nlargest(3, "mi_r2_ratio")[["mi", "r2", "mi_r2_ratio"]]
    return "\n".join([
        "# Specification diagnostics (prepared train)",
        "",
        "Ramsey RESET · Pearson-vs-Spearman discrepancies · MI-vs-R² audit.",
        "",
        f"- Rows: {n_rows:,} (prepared train, NaN-dropped); target: {TARGET}.",
        f"- RESET baseline: {TARGET} ~ {' + '.join(RESET_X_COLS)} (plain OLS).",
        f"- RESET power=2 (yhat²): F={f2:,.1f}, p={p2:.3e} → {verdict2}.",
        f"- RESET power=3 (yhat²+yhat³): F={f3:,.1f}, p={p3:.3e} → {verdict3}.",
        "",
        "## RESET variants (non-linear monotonic trap — duration curvature)",
        "",
        ("trip_duration_minutes is monotonic but curved; log or quadratic terms "
         "should collapse the specification error."),
        "",
        _markdown_table(variants),
        "",
        "## Evidence table",
        "",
        _markdown_table(evidence),
        "",
        f"## Flagged: top-3 |Pearson − Spearman| (threshold Δ ≥ {DELTA_THRESHOLD:.2f})",
        "",
        (f"Features over threshold: {', '.join(delta_flagged)}." if delta_flagged
         else "No feature exceeds the threshold."),
        "",
        _markdown_table(top_delta),
        "",
        "## Flagged: top-3 MI/R² ratio (nonlinearity candidates)",
        "",
        _markdown_table(top_ratio),
        "",
    ])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PREPARED_TRAIN).dropna(subset=[*AUDIT_COLS, TARGET]).reset_index(drop=True)
    print(f"prepared train rows: {len(df):,}")

    model = fit_linear_baseline(df)
    reset = run_reset(model)
    variants = run_reset_variants(df)

    evidence = correlation_evidence(df).join(mi_r2_evidence(df))
    print(evidence.round(4).to_string())
    evidence.to_csv(DESCRIBE_CSV)
    print(f"wrote {DESCRIBE_CSV} ({len(evidence)} rows)")

    plot_main(evidence, PNG_MAIN)
    print(f"wrote {PNG_MAIN}")

    plot_reset(model, reset, df, PNG_RESET)
    print(f"wrote {PNG_RESET}")

    SUMMARY_MD.write_text(_summary_md(len(df), evidence, reset, variants))
    print(f"wrote {SUMMARY_MD}")

    top_delta = evidence.nlargest(3, "delta").index.tolist()
    top_ratio = evidence.nlargest(3, "mi_r2_ratio").index.tolist()
    print(f"top-3 |delta| features: {top_delta}")
    print(f"top-3 MI/R2 ratio features: {top_ratio}")


if __name__ == "__main__":
    main()
