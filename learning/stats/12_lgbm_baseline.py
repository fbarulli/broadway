"""
12_lgbm_baseline.py

The non-linear baseline promised after the OLS diagnostics made the case
against pure linear modeling: heteroskedastic residuals (Breusch-Pagan
p=0), heavy right tail (kurtosis=25.5), and now (11) likely borough-varying
slopes. Trees don't assume homoskedastic normal errors or a fixed linear
functional form, so they should absorb all three of those issues without
needing a workaround the way OLS does.

Two things this script does differently from the OLS scripts, deliberately:

  1. TIME-BASED split, not random. Taxi demand/traffic is autocorrelated
     (see 10 -- pickup_datetime carries structure). A random train/test
     split would let the model see trips from the same rush-hour window
     it's being tested on, leaking information and overstating performance.
     Train on everything before a cutoff date, test on everything after.

  2. Uses the full ENGINEERED_FEATURES set from features/, not just
     distance + borough -- this is meant as a real baseline to build on,
     not another diagnostic-only fit.

Compares against the OLS baseline (08) and interaction model (11) on the
same holdout using MAE and RMSE, since R^2 alone hides how the model does
in the heavy tail that was the whole point of this diagnostic detour.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from logistics_ml.features.schema import ENGINEERED_FEATURES, TARGET
from logistics_ml.features.pipeline import FeaturePipeline

TIME_SPLIT_CUTOFF = "2024-06-01"  # adjust to match actual data coverage


def load_and_split():
    df = pd.read_parquet("data/processed/training_data.parquet")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.sort_values("pickup_datetime")

    train = df[df["pickup_datetime"] < TIME_SPLIT_CUTOFF]
    test = df[df["pickup_datetime"] >= TIME_SPLIT_CUTOFF]

    print(f"Train: {len(train)} rows (< {TIME_SPLIT_CUTOFF})")
    print(f"Test:  {len(test)} rows (>= {TIME_SPLIT_CUTOFF})")

    pipeline = FeaturePipeline()
    pipeline.fit(train)
    train_feat = pipeline.transform(train)
    test_feat = pipeline.transform(test)

    return train_feat, test_feat


def train_lgbm(train_feat: pd.DataFrame):
    X_train = train_feat[ENGINEERED_FEATURES]
    y_train = train_feat[TARGET] if TARGET in train_feat else None

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, test_feat: pd.DataFrame) -> dict:
    X_test = test_feat[ENGINEERED_FEATURES]
    y_test = test_feat[TARGET]
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Tail-specific check: this is the part OLS struggled with
    # (kurtosis=25.5 in 08's residuals). MAE/RMSE on the slowest decile
    # tells you directly whether the tree model actually fixed that,
    # rather than just improving the easy majority of trips.
    tail_mask = y_test >= y_test.quantile(0.9)
    tail_mae = mean_absolute_error(y_test[tail_mask], preds[tail_mask])

    return {"mae": mae, "rmse": rmse, "tail_mae_p90": tail_mae}


def main():
    print("Loading data with time-based split...")
    train_feat, test_feat = load_and_split()

    print("\nTraining LightGBM baseline...")
    model = train_lgbm(train_feat)

    print("\n=== Holdout performance (time-based split) ===")
    metrics = evaluate(model, test_feat)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print("\n=== Feature importance (top 15) ===")
    importance = pd.Series(
        model.feature_importances_, index=ENGINEERED_FEATURES
    ).sort_values(ascending=False)
    print(importance.head(15))

    print(
        "\nCompare mae/rmse here against 08 (baseline OLS) and 11 "
        "(interaction OLS) computed on the SAME time-based holdout -- "
        "the OLS scripts as written use random stratified samples, so "
        "for a fair comparison, rerun their .predict() on this script's "
        "test_feat slice rather than comparing R^2 across different "
        "samples/splits directly."
    )


if __name__ == "__main__":
    main()