import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    clean_df = df[[feature, target]].dropna()
    y = clean_df[target].to_numpy()
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    
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
    omega_num = ss_between - (df_between * ms_within)
    omega_den = ss_total + ms_within
    omega_sq = omega_num / omega_den if omega_den > 0 else 0.0
    
    return {
        "feature": feature, "k_groups": k, "n_total": n,
        "f_stat": f_stat, "p_value": p_value,
        "eta_squared": eta_sq, "omega_squared": omega_sq,
    }

def plot_effect_sizes(out: pd.DataFrame, out_path: Path) -> None:
    plot_df = out.set_index("feature")[["eta_squared", "omega_squared", "f_stat", "p_value"]]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = np.arange(len(plot_df))
    width = 0.35
    
    ax.bar(x - width/2, plot_df["eta_squared"], width, label=r"Eta-squared ($\eta^2$)", color="#4c72b0")
    ax.bar(x + width/2, plot_df["omega_squared"], width, label=r"Omega-squared ($\omega^2$)", color="#d62728")
    
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        eta2, omega2 = row["eta_squared"], row["omega_squared"]
        f_stat, p_val = row["f_stat"], row["p_value"]
        
        ax.text(x[i] - width/2, eta2 + 0.005, f"{eta2:.3f}", ha="center", va="bottom", fontsize=10, color="#4c72b0", fontweight="bold")
        ax.text(x[i] + width/2, omega2 + 0.005, f"{omega2:.3f}", ha="center", va="bottom", fontsize=10, color="#d62728", fontweight="bold")
        
        p_text = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
        ax.text(x[i], max(eta2, omega2) + 0.03, f"F={f_stat:,.0f}\n{p_text}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("Effect Size (Proportion of Variance Explained)")
    ax.set_title("Univariate ANOVA: Location Zones vs Fare Amount")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def summary_md(out: pd.DataFrame) -> str:
    lines = [
        r"# Categorical ANOVA: Location Zones vs Fare Amount",
        "",
        f"One-way ANOVA testing the variance in `{TARGET}` explained by categorical zone IDs.",
        "",
        r"- **$\eta^2$ (Eta-squared):** The proportion of total variance in the fare explained by the zone.",
        r"- **$\omega^2$ (Omega-squared):** A less biased estimate of the population effect size.",
        "",
        "## Results", "",
        "| Feature | Zones (k) | F-statistic | p-value | η² | ω² |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(f"| {row['feature']} | {int(row['k_groups'])} | {row['f_stat']:,.2f} | {row['p_value']:.2e} | {row['eta_squared']:.4f} | {row['omega_squared']:.4f} |")
    return "\n".join(lines)

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    rows = [compute_anova(df, feat, TARGET) for feat in CATEGORICAL_FEATURES]
    out = pd.DataFrame(rows)
    
    out.to_csv(CSV_OUT)
    print(out[["feature", "k_groups", "f_stat", "p_value", "eta_squared", "omega_squared"]].to_string(index=False))
    plot_effect_sizes(out, Path(PNG_OUT))
    MD_OUT.write_text(summary_md(out))
    print(f"wrote {CSV_OUT}, {PNG_OUT}, {MD_OUT}")

if __name__ == "__main__":
    main()
