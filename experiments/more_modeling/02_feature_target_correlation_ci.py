"""02: confidence intervals for feature-target correlation.

Evidence-only: computes a 95% CI for the Pearson correlation between every
numeric column and the target, using the Fisher z-transformation.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

from _common import RESULTS, TARGET, load_sample, numeric_features

CONFIDENCE = 0.95
Z_95 = 1.959963984540054
CSV_OUT = RESULTS / "02_feature_target_correlation_ci.csv"
PNG_OUT = RESULTS / "02_feature_target_correlation_ci.png"
MD_OUT = RESULTS / "02_feature_target_correlation_ci.md"


def correlation_ci(
    feature: str, x: pd.Series, y: pd.Series, z: float = Z_95
) -> dict[str, object]:
    """Compute Pearson r and its confidence interval via Fisher z."""
    valid_idx = x.index[
        np.isfinite(x) & np.isfinite(y) & ~x.isna() & ~y.isna()
    ]
    x_clean = x.loc[valid_idx]
    y_clean = y.loc[valid_idx]
    n = len(x_clean)

    row = {
        "feature": feature,
        "n": n,
        "confidence": CONFIDENCE,
        "method": "pearson_fisher_z",
    }

    if n < 4:
        row.update(
            {
                "r": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "warning": "too few valid observations",
            }
        )
        return row

    r, p_value = stats.pearsonr(x_clean, y_clean)
    
    z_r = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)

    z_low = z_r - z * se_z
    z_high = z_r + z * se_z
    
    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)

    row.update(
        {
            "r": r,
            "p_value": p_value,
            "ci_low": r_low,
            "ci_high": r_high,
            "warning": "",
        }
    )
    return row


def plot_correlation_ci(out: pd.DataFrame, out_path: Path) -> None:
    """Horizontal barchart of correlation coefficients with 95% CI error bars."""
    plot_df = out.dropna(subset=["r"]).copy()
    plot_df = plot_df.sort_values(by="r", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.6)), constrained_layout=True)
    
    sns.barplot(
        x=plot_df["r"],
        y=plot_df.index,
        ax=ax,
        color="#4c72b0",
    )
    
    xerr_low = plot_df["r"] - plot_df["ci_low"]
    xerr_high = plot_df["ci_high"] - plot_df["r"]
    
    ax.errorbar(
        x=plot_df["r"],
        y=range(len(plot_df)),
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="#d62728",
        elinewidth=1.5,
        capsize=3,
    )
    
    # Add exact numbers at the end of each bar
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        r_val = row["r"]
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        text = f"r = {r_val:.3f}  [{ci_low:.3f}, {ci_high:.3f}]"
        ax.text(
            ci_high + 0.015, 
            i, 
            text, 
            va="center", 
            ha="left", 
            fontsize=9,
            color="#333333"
        )
        
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson correlation (r) with {TARGET}")
    ax.set_ylabel("")
    
    n_val = int(plot_df["n"].iloc[0]) if not plot_df.empty else 0
    ax.set_title(f"95% Confidence Intervals for Feature-Target Correlation (N={n_val:,})")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Explicitly set x-limits to leave room for the text annotations
    x_min = min(-0.2, plot_df["ci_low"].min() - 0.02)
    x_max = plot_df["ci_high"].max() + 0.25
    ax.set_xlim(x_min, x_max)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_md(out: pd.DataFrame) -> str:
    """Render the correlation results as a Markdown table."""
    lines = [
        "# Feature-Target Correlation Confidence Intervals",
        "",
        f"95% Confidence Intervals for the Pearson correlation between numeric features and `{TARGET}` (N={int(out['n'].iloc[0]):,}).",
        "",
        "Computed using the Fisher z-transformation. Note the extremely tight intervals due to the large sample size.",
        "",
        "## Results",
        "",
        "| Feature | r | 95% CI Low | 95% CI High | p-value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idx, row in out.iterrows():
        lines.append(
            f"| {idx} | {row['r']:.4f} | {row['ci_low']:.4f} | {row['ci_high']:.4f} | {row['p_value']:.2e} |"
        )
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    features = numeric_features(df)
    target = df[TARGET]

    rows = [correlation_ci(feat, df[feat], target) for feat in features]
    out = pd.DataFrame(rows).set_index("feature")
    out.to_csv(CSV_OUT)

    print(f"features: {len(out)}")
    print(out[["r", "ci_low", "ci_high", "p_value"]].to_string())
    print(f"wrote {CSV_OUT}")

    plot_correlation_ci(out, Path(PNG_OUT))
    print(f"wrote {PNG_OUT}")

    MD_OUT.write_text(summary_md(out))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
