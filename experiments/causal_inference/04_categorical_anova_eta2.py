"""04: ANOVA and effect size (Eta-squared / Omega-squared) for categorical features.

Evaluates how much of the total variance in the target is explained by 
the categorical features (pickup/dropoff zones), ignoring confounders.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats

from _common import RESULTS, TARGET, load_sample

CATEGORICAL_FEATURES = ["pickup_location_id", "dropoff_location_id"]
CONFIDENCE = 0.95

CSV_OUT = RESULTS / "04_categorical_anova_eta2.csv"
PNG_OUT = RESULTS / "04_categorical_anova_eta2.png"
MD_OUT = RESULTS / "04_categorical_anova_eta2.md"


def compute_anova(df: pd.DataFrame, feature: str, target: str) -> dict[str, object]:
    """Vectorized one-way ANOVA computing F-stat, p-value, eta², and omega²."""
    clean_df = df[[feature, target]].dropna()
    y = clean_df[target].to_numpy()
    
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    
    # Vectorized group means
    group_means = clean_df.groupby(feature)[target].transform("mean").to_numpy()
    ss_between = np.sum((group_means - grand_mean) ** 2)
    ss_within = ss_total - ss_between
    
    k = clean_df[feature].nunique()
    n = len(clean_df)
    
    df_between = k - 1
    df_within = n - k
    
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    
    f_stat = ms_between / ms_within if ms_within > 0 else np.nan
    p_value = stats.f.sf(f_stat, df_between, df_within) if not np.isnan(f_stat) else np.nan
    
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    
    # Omega-squared (corrects for sample bias)
    omega_num = ss_between - (df_between * ms_within)
    omega_den = ss_total + ms_within
    omega_sq = omega_num / omega_den if omega_den > 0 else 0.0
    
    return {
        "feature": feature,
        "k_groups": k,
        "n_total": n,
        "f_stat": f_stat,
        "p_value": p_value,
        "eta_squared": eta_sq,
        "omega_squared": omega_sq,
    }


def plot_effect_sizes(out: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart comparing Eta-squared and Omega-squared."""
    plot_df = out.set_index("feature")[["eta_squared", "omega_squared"]]
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    x = np.arange(len(plot_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, plot_df["eta_squared"], width, label="Eta-squared (η²)", color="#4c72b0")
    bars2 = ax.bar(x + width/2, plot_df["omega_squared"], width, label="Omega-squared (ω²)", color="#d62728")
    
    # Add text labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height + 0.002,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=10
            )
            
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("Effect Size (Proportion of Variance Explained)")
    ax.set_title("Variance in Fare Amount Explained by Location Zones (Univariate)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_md(out: pd.DataFrame) -> str:
    """Render the ANOVA results as a Markdown table."""
    lines = [
        "# Categorical ANOVA: Location Zones vs Fare Amount",
        "",
        f"One-way ANOVA testing the variance in `{TARGET}` explained by categorical zone IDs.",
        "",
        "- **$\eta^2$ (Eta-squared):** The proportion of total variance in the fare explained by the zone.",
        "- **$\omega^2$ (Omega-squared):** A less biased estimate of the population effect size.",
        "",
        "## Results",
        "",
        "| Feature | Zones (k) | F-statistic | p-value | η² | ω² |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['k_groups'])} | {row['f_stat']:,.2f} | {row['p_value']:.2e} | {row['eta_squared']:.4f} | {row['omega_squared']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    
    print("Computing ANOVA for categorical features...")
    rows = [compute_anova(df, feat, TARGET) for feat in CATEGORICAL_FEATURES]
    out = pd.DataFrame(rows)
    
    out.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")
    
    print(out[["feature", "k_groups", "f_stat", "p_value", "eta_squared", "omega_squared"]].to_string(index=False))

    plot_effect_sizes(out, Path(PNG_OUT))
    print(f"wrote {PNG_OUT}")

    MD_OUT.write_text(summary_md(out))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
