"""05: The Joint Model (Numeric + Top-10 Categorical Zones).

Combines numeric features (distance, duration) with the top 10 most frequent
pickup zones. The most frequent zone is used as the baseline reference.
The model outputs the isolated dollar premium for each zone, holding trip
distance and duration perfectly constant.

Encoding note: zones are one-hot encoded by a sklearn ColumnTransformer fit on
the modeling partition only. A zone unseen at fit time encodes as an all-zeros
row under ``handle_unknown="ignore"`` — i.e. it is treated as the reference
level's effect. This is a stated modeling assumption for the causal reading of
the results.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from _common import INDEPENDENT_NUMERIC_FEATURES, RESULTS, TARGET, load_sample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CONFIDENCE = 0.95

CSV_OUT = RESULTS / "05_joint_model_coefs.csv"
PNG_OUT = RESULTS / "05_zone_premiums.png"
MD_OUT = RESULTS / "05_joint_model_summary.md"


def prepare_joint_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """Prepare numeric features + top-10 zone dummies via a fit-on-train encoder."""
    num_cols = INDEPENDENT_NUMERIC_FEATURES
    cat_col = "pickup_location_id"
    
    valid = df[num_cols + [cat_col, TARGET]].dropna()
    y = valid[TARGET]
    
    zone_counts = valid[cat_col].value_counts()
    top_10_zones = zone_counts.head(10).index.tolist()
    baseline_zone = str(top_10_zones[0])  # Most frequent zone
    
    # Non-top-10 zones consolidate into "Other" before encoding.
    mapped = valid[cat_col].map(lambda z: str(z) if z in top_10_zones else "Other")
    raw = valid[num_cols].copy()
    raw["zone"] = mapped
    
    # OneHotEncoder fit on the modeling partition only; unseen zones at
    # predict time encode as all-zeros (reference-level effect).
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None,
                                 sparse_output=False), ["zone"]),
    ])
    pipeline = Pipeline([("pre", pre)])
    encoded = pipeline.fit_transform(raw)
    columns = [name.split("__", 1)[1] for name in pipeline.get_feature_names_out()]
    X = pd.DataFrame(encoded, index=raw.index, columns=columns)
    
    # Drop the baseline zone to prevent the dummy variable trap
    baseline_col = f"zone_{baseline_zone}"
    if baseline_col in X.columns:
        X = X.drop(columns=[baseline_col])
        
    X = sm.add_constant(X)
    
    return X, y, baseline_zone


def plot_zone_premiums(results_df: pd.DataFrame, baseline_zone: str, out_path: Path) -> None:
    """Bar chart of the dollar premium for top zones vs baseline."""
    zone_coefs = results_df[results_df.index.str.startswith("zone_")].copy()
    
    if zone_coefs.empty:
        print("No zone coefficients to plot.")
        return

    zone_coefs["zone_name"] = zone_coefs.index.str.replace("zone_", "Zone ")
    zone_coefs = zone_coefs.sort_values("raw_coef_dollars", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    sns.barplot(
        x=zone_coefs["raw_coef_dollars"],
        y=zone_coefs["zone_name"],
        ax=ax,
        color="#4c72b0"
    )
    
    xerr_low = zone_coefs["raw_coef_dollars"] - zone_coefs["ci_low"]
    xerr_high = zone_coefs["ci_high"] - zone_coefs["raw_coef_dollars"]
    
    ax.errorbar(
        x=zone_coefs["raw_coef_dollars"],
        y=range(len(zone_coefs)),
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor="#d62728",
        elinewidth=1.5,
        capsize=3,
    )
    
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Zone Premium ($ vs Baseline, holding distance/time constant)")
    ax.set_ylabel("Pickup Zone")
    ax.set_title(f"Isolated Zone Premiums (Baseline: Zone {baseline_zone})")
    ax.grid(True, alpha=0.3, axis="x")
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_md(results_df: pd.DataFrame, baseline_zone: str, r_squared: float) -> str:
    """Render the joint model results."""
    num_rows = results_df.loc[INDEPENDENT_NUMERIC_FEATURES]
    zone_rows = results_df[results_df.index.str.startswith("zone_")].sort_values("raw_coef_dollars", ascending=False)
    
    lines = [
        "# The Joint Model: Numeric + Top-10 Pickup Zones",
        "",
        f"Multivariate OLS predicting `{TARGET}`. The most frequent zone (Zone {baseline_zone}) is the baseline reference.",
        "",
        f"**R-squared:** {r_squared:.4f}",
        "",
        "## Numeric Coefficients (Dollars per unit)",
        "",
        "| Feature | Coefficient ($/unit) | 95% CI Low | 95% CI High | p-value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for feat, row in num_rows.iterrows():
        lines.append(f"| {feat} | ${row['raw_coef_dollars']:.4f} | ${row['ci_low']:.4f} | ${row['ci_high']:.4f} | {row['p_value']:.2e} |")
        
    lines.extend([
        "",
        f"## Isolated Zone Premiums (Dollars vs Zone {baseline_zone})",
        "",
        "| Zone | Premium ($) | 95% CI Low | 95% CI High | p-value |",
        "| --- | --- | --- | --- | --- |",
    ])
    for feat, row in zone_rows.iterrows():
        zone_name = feat.replace("zone_", "Zone ")
        lines.append(f"| {zone_name} | ${row['raw_coef_dollars']:.4f} | ${row['ci_low']:.4f} | ${row['ci_high']:.4f} | {row['p_value']:.2e} |")
        
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    X, y, baseline_zone = prepare_joint_data(df)
    
    print(f"Fitting Joint Model on {len(X):,} observations...")
    print(f"Baseline reference: Zone {baseline_zone}")
    
    model = sm.OLS(y, X).fit()
    
    conf_int = model.conf_int(alpha=1 - CONFIDENCE)
    results_df = pd.DataFrame({
        "raw_coef_dollars": model.params.values,
        "ci_low": conf_int[0].values,
        "ci_high": conf_int[1].values,
        "p_value": model.pvalues.values
    }, index=model.params.index)
    
    results_df.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")
    
    print("\n--- Numeric Coefficients ---")
    print(results_df.loc[INDEPENDENT_NUMERIC_FEATURES][["raw_coef_dollars", "p_value"]].to_string())
    
    print("\n--- Top Zone Premiums ---")
    zone_rows = results_df[results_df.index.str.startswith("zone_")].sort_values("raw_coef_dollars", ascending=False)
    print(zone_rows[["raw_coef_dollars", "p_value"]].head(5).to_string())

    plot_zone_premiums(results_df, baseline_zone, Path(PNG_OUT))
    print(f"wrote {PNG_OUT}")

    MD_OUT.write_text(summary_md(results_df, baseline_zone, model.rsquared))
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
