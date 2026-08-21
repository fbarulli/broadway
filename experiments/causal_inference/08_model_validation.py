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
from sklearn.model_selection import train_test_split

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

CSV_OUT = RESULTS / "08_model_validation_summary.csv"
PNG_COEFS = RESULTS / "08_coefficient_stability.png"
PNG_BUCKETS = RESULTS / "08_fare_bucket_residuals.png"
MD_OUT = RESULTS / "08_model_validation.md"

# Fare buckets for Step 3 audit
FARE_BINS = [0, 10, 20, 30, 50, 100, np.inf]
FARE_LABELS = ['$0-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()
    
    # --- Step 1: Train-Test Split & Consistency Check ---
    print("Splitting data 80/20...")
    train_df, test_df = train_test_split(
        df[INDEPENDENT_NUMERIC_FEATURES + [TARGET]].dropna(), 
        test_size=0.20, random_state=42
    )
    
    X_train, y_train = train_df[INDEPENDENT_NUMERIC_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[INDEPENDENT_NUMERIC_FEATURES], test_df[TARGET]
    
    X_train = sm.add_constant(X_train)
    X_test_full = sm.add_constant(X_test) # Used for prediction
    
    model_train = sm.OLS(y_train, X_train).fit()
    
    # Fit model on TEST data purely to check coefficient stability
    X_test_for_fit = sm.add_constant(X_test)
    model_test = sm.OLS(y_test, X_test_for_fit).fit()
    
    train_coefs = pd.Series(model_train.params, name="Train (80%)")
    test_coefs = pd.Series(model_test.params, name="Test (20%)")
    coef_df = pd.concat([train_coefs, test_coefs], axis=1)
    coef_df["diff"] = coef_df["Train (80%)"] - coef_df["Test (20%)"]
    
    print("\n--- Step 1: Coefficient Stability ---")
    print(coef_df.round(4).to_string())
    
    # --- Step 2: Out-of-Sample Predictive Testing (MAPE) ---
    preds = model_train.predict(X_test_full)
    
    # MAPE explodes near $0 (e.g., $0.50 fare with $1.00 error = 100% error). 
    # Filter actuals > $1.00 for a realistic business MAPE.
    valid_mask = y_test >= 1.0
    mape = np.mean(np.abs((y_test[valid_mask] - preds[valid_mask]) / y_test[valid_mask])) * 100
    mae = np.mean(np.abs(y_test - preds))
    r2 = model_train.rsquared
    
    print(f"\n--- Step 2: Test Set Performance ---")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE (fares > $1): {mape:.2f}%")
    
    # --- Step 3: Multi-Segment Outlier & Residual Audit ---
    eval_df = pd.DataFrame({"actual": y_test, "pred": preds})
    eval_df["error"] = eval_df["actual"] - eval_df["pred"]
    eval_df["bucket"] = pd.cut(eval_df["actual"], bins=FARE_BINS, labels=FARE_LABELS)
    
    bucket_stats = eval_df.groupby("bucket", observed=False).agg(
        count=("error", "size"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        mae=("error", lambda x: np.abs(x).mean())
    ).reset_index()
    
    print("\n--- Step 3: Fare Bucket Residual Audit ---")
    print(bucket_stats.round(2).to_string(index=False))
    
    # --- Save CSVs and Markdown ---
    summary_df = pd.concat([
        coef_df.T,
        pd.DataFrame({"metric": ["MAE", "MAPE_pct", "R2"], "Train (80%)": [mae, mape, r2], "Test (20%)": [np.nan, np.nan, np.nan]}).set_index("metric")
    ])
    summary_df.to_csv(CSV_OUT)
    
    md_lines = [
        "# Model Validation Audit",
        "",
        f"Evaluation of the `{TARGET}` linear model using an out-of-sample 20% holdout set.",
        "",
        "## Step 1: Coefficient Stability",
        "If the formulas reflect true physics, they must remain identical on unseen data.",
        "",
        coef_df.round(4).to_markdown(),
        "",
        "## Step 2: Out-of-Sample Predictive Testing",
        f"- **MAE:** ${mae:.2f}",
        f"- **MAPE (fares > $1):** {mape:.2f}%",
        f"- **R-squared:** {r2:.4f}",
        "",
        "## Step 3: Multi-Segment Residual Audit",
        "Checking for non-linearity. If the model is perfectly linear, the mean error per bucket should hover near $0.00.",
        "",
        bucket_stats.round(2).to_markdown(index=False)
    ]
    MD_OUT.write_text("\n".join(md_lines))
    
    # --- Plotting ---
    fig1, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    coef_plot = coef_df.drop("const").copy()
    x = np.arange(len(coef_plot))
    width = 0.35
    ax1.bar(x - width/2, coef_plot["Train (80%)"], width, label="Train", color="#4c72b0")
    ax1.bar(x + width/2, coef_plot["Test (20%)"], width, label="Test", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(coef_plot.index)
    ax1.set_ylabel("Coefficient ($/unit)")
    ax1.set_title("Step 1: Train vs Test Coefficient Stability")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.savefig(PNG_COEFS, dpi=150)
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=bucket_stats, x="bucket", y="mean_error", ax=ax2, color="#4c72b0")
    ax2.axhline(0, color="#d62728", linestyle="--", lw=1.5)
    ax2.set_xlabel("Actual Fare Bucket")
    ax2.set_ylabel("Mean Error (Actual - Predicted)")
    ax2.set_title("Step 3: Mean Residual by Fare Bucket (Non-linearity Check)")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add counts on bars
    for i, row in bucket_stats.iterrows():
        ax2.text(i, row["mean_error"] + 0.2, f"n={row['count']:,}", ha="center", fontsize=9)
        
    fig2.savefig(PNG_BUCKETS, dpi=150)
    plt.close(fig2)
    
    print(f"\nwrote {CSV_OUT}, {PNG_COEFS}, {PNG_BUCKETS}, {MD_OUT}")


if __name__ == "__main__":
    main()
