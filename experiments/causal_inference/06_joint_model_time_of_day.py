import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

CSV_OUT = RESULTS / "06_time_of_day_coefs.csv"
PNG_OUT = RESULTS / "06_time_of_day_premiums.png"
MD_OUT = RESULTS / "06_time_of_day_summary.md"

BASELINE_HOUR = 12  # Noon

def prepare_time_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Extract hour if not already in the dataframe
    if "pickup_hour" not in df.columns:
        if "pickup_datetime" in df.columns:
            df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour
        else:
            raise ValueError("Need pickup_datetime or pickup_hour to run time-of-day model.")
            
    valid = df[INDEPENDENT_NUMERIC_FEATURES + ["pickup_hour", TARGET]].dropna()
    X_num = valid[INDEPENDENT_NUMERIC_FEATURES]
    y = valid[TARGET]
    
    # Create hour dummies, dropping the baseline hour to prevent dummy variable trap
    hour_dummies = pd.get_dummies(valid["pickup_hour"], prefix="hour", dtype=int)
    baseline_col = f"hour_{BASELINE_HOUR}"
    if baseline_col in hour_dummies.columns:
        hour_dummies = hour_dummies.drop(columns=[baseline_col])
        
    X = pd.concat([X_num, hour_dummies], axis=1)
    X = sm.add_constant(X)
    return X, y

def plot_time_premiums(results_df: pd.DataFrame, out_path: Path) -> None:
    hour_coefs = results_df[results_df.index.str.startswith("hour_")].copy()
    
    # Extract hour integer for proper chronological sorting
    hour_coefs["hour_int"] = hour_coefs.index.str.replace("hour_", "").astype(int)
    hour_coefs = hour_coefs.sort_values("hour_int")
    hour_coefs["hour_label"] = hour_coefs["hour_int"].apply(lambda h: f"{h:02d}:00")
    
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    
    sns.barplot(x=hour_coefs["hour_label"], y=hour_coefs["raw_coef_dollars"], ax=ax, color="#4c72b0")
    
    xerr_low = hour_coefs["raw_coef_dollars"] - hour_coefs["ci_low"]
    xerr_high = hour_coefs["ci_high"] - hour_coefs["raw_coef_dollars"]
    ax.errorbar(x=range(len(hour_coefs)), y=hour_coefs["raw_coef_dollars"], 
                yerr=[xerr_low, xerr_high], fmt="none", ecolor="#d62728", capsize=3)
                
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pickup Hour")
    ax.set_ylabel(f"Premium ($) vs {BASELINE_HOUR}:00 (holding distance/time constant)")
    ax.set_title(f"Isolated Time-of-Day Fare Premiums (Baseline: {BASELINE_HOUR}:00 Noon)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    X, y = prepare_time_data(df)
    
    print(f"Fitting Time-of-Day Joint Model on {len(X):,} observations...")
    model = sm.OLS(y, X).fit()
    
    conf_int = model.conf_int(alpha=0.05)
    results_df = pd.DataFrame({
        "raw_coef_dollars": model.params.values,
        "ci_low": conf_int[0].values,
        "ci_high": conf_int[1].values,
        "p_value": model.pvalues.values
    }, index=model.params.index)
    
    results_df.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")
    
    print("\n--- Numeric Meter Rates (controlling for time of day) ---")
    print(results_df.loc[INDEPENDENT_NUMERIC_FEATURES][["raw_coef_dollars", "p_value"]].to_string())
    
    print(f"\n--- Top Time Premiums (vs {BASELINE_HOUR}:00 Noon) ---")
    hour_rows = results_df[results_df.index.str.startswith("hour_")].sort_values("raw_coef_dollars", ascending=False)
    print(hour_rows[["raw_coef_dollars", "p_value"]].head(5).to_string())

    plot_time_premiums(results_df, Path(PNG_OUT))
    print(f"wrote {PNG_OUT}")

if __name__ == "__main__":
    main()
