"""03: confidence intervals for feature regression effects.

Causal/Statistical core: fits a multivariate OLS regression to estimate the
isolated effect of each feature on the target, holding other features constant.
Features are standardized (z-scored) so coefficients represent the change in
the target (in standard deviations) per 1 standard deviation increase in the
feature.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

CONFIDENCE = 0.95
CSV_OUT = RESULTS / "03_feature_effect_ci.csv"
PNG_OUT = RESULTS / "03_feature_effect_ci.png"
MD_OUT = RESULTS / "03_feature_effect_ci.md"


def clean_and_scale(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Drop inf/nan and standardize features to mean=0, std=1."""
    X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Subset y to match the cleaned X index
    y = df.loc[X.index, TARGET]
    
    # Standardize X
    X_scaled = (X - X.mean()) / X.std()
    
    # Add constant for intercept
    X_scaled = sm.add_constant(X_scaled)
    
    return X_scaled, y


def plot_effects(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot standardized regression coefficients with 95% CI error bars."""
    # Drop the intercept for plotting
    plot_df = results_df.drop(index="const").sort_values(by="coef", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.6)), constrained_layout=True)
    
    sns.barplot(
        x=plot_df["coef"],
        y=plot_df.index,
        ax=ax,
        color="#4c72b0",
    )
    
    xerr_low = plot_df["coef"] - plot_df["ci_low"]
    xerr_high = plot_df["ci_high"] - plot_df["coef"]
    
    ax.errorbar(
        x=plot_df["coef"],
        y=range(len(plot_df)),
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="#d62728",
        elinewidth=1.5,
        capsize=3,
    )
    
    # Add exact numbers at the end of each bar
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        coef = row["coef"]
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        text = f"β = {coef:.3f}  [{ci_low:.3f}, {ci_high:.3f}]"
        
        # Place text on the right or left depending on the bar direction
        ha = "left" if coef >= 0 else "right"
        x_pos = ci_high + 0.02 if coef >= 0 else ci_low - 0.02
        
        ax.text(x_pos, i, text, va="center", ha=ha, fontsize=9, color="#333333")
        
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Regression Coefficient (β)")
    ax.set_ylabel("")
    ax.set_title(f"95% Confidence Intervals for Isolated Feature Effects on {TARGET}")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Give room for text
    max_abs = max(abs(plot_df["ci_low"].min()), abs(plot_df["ci_high"].max()))
    ax.set_xlim(-max_abs - 0.3, max_abs + 0.3)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_md(results_df: pd.DataFrame) -> str:
    """Render the regression results as a Markdown table."""
    lines = [
        "# Isolated Feature Effect Confidence Intervals (Multivariate OLS)",
        "",
        f"95% Confidence Intervals for standardized regression coefficients (β) predicting `{TARGET}`.",
        "",
        "Features are z-scored (mean=0, std=1) before fitting. A β of 0.5 means a 1 standard deviation increase in the feature is associated with a 0.5 standard deviation increase in the fare, *holding all other features constant*.",
        "",
        "## Results",
        "",
        "| Feature | β (Coef) | 95% CI Low | 95% CI High | p-value |",
        "| --- | --- | --- | --- | --- |",
    ]
    
    # Exclude intercept from the main table, or put it at the bottom
    features_only = results_df.drop(index="const")
    for idx, row in features_only.iterrows():
        lines.append(
            f"| {idx} | {row['coef']:.4f} | {row['ci_low']:.4f} | {row['ci_high']:.4f} | {row['p_value']:.2e} |"
        )
        
    lines.extend([
        "",
        f"**Model Fit:** R-squared = {results_df.attrs.get('r_squared', 'N/A')}",
        f"**Observations:** {int(results_df.attrs.get('n_obs', 0)):,}",
    ])
    
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    features = INDEPENDENT_NUMERIC_FEATURES
    
    print("Cleaning and scaling features...")
    X_scaled, y = clean_and_scale(df, features)
    
    print(f"Fitting OLS on {len(X_scaled):,} observations...")
    model = sm.OLS(y, X_scaled).fit()
    
    # Extract coefficients and CIs
    conf_int = model.conf_int(alpha=1 - CONFIDENCE)
    
    results_df = pd.DataFrame({
        "coef": model.params,
        "ci_low": conf_int[0],
        "ci_high": conf_int[1],
        "p_value": model.pvalues,
    })
    
    # Store model stats in attrs for the markdown summary
    results_df.attrs["r_squared"] = f"{model.rsquared:.4f}"
    results_df.attrs["n_obs"] = model.nobs
    
    results_df.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

    plot_effects(results_df, Path(PNG_OUT))
    print(f"wrote {PNG_OUT}")

    MD_OUT.write_text(summary_md(results_df))
    print(f"wrote {MD_OUT}")
    
    print("\nTop effects (absolute magnitude):")
    plot_df = results_df.drop(index="const").copy()
    plot_df["abs_coef"] = plot_df["coef"].abs()
    print(plot_df.sort_values("abs_coef", ascending=False)[["coef", "ci_low", "ci_high"]].to_string())


if __name__ == "__main__":
    main()
