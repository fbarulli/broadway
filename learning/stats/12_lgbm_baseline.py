import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from broadway.features.schema import ENGINEERED_FEATURES, TARGET
from broadway.features.ml_pipeline import FeaturePipeline

from _config import (
    DATA_PATH,
    TIME_SPLIT_CUTOFF,
    SAMPLE_SIZE,
    RANDOM_STATE,
    DATETIME_COL,
    FEATURE_LOOKUP_PATH,
    FEATURE_ENCODING_SMOOTHING,
    FEATURE_FREQUENCY_FILL,
    FEATURE_RUSH_HOUR_MORNING_START,
    FEATURE_RUSH_HOUR_MORNING_END,
    FEATURE_RUSH_HOUR_EVENING_START,
    FEATURE_RUSH_HOUR_EVENING_END,
    FEATURE_NIGHT_START,
    FEATURE_NIGHT_END,
    FEATURE_PASSENGER_COUNT_MIN,
    FEATURE_PASSENGER_COUNT_MAX,
    N_ESTIMATORS,
    LEARNING_RATE,
    NUM_LEAVES,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    QUANTILE_TAIL,
)


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_PATH)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL)

    train = df[df[DATETIME_COL] < TIME_SPLIT_CUTOFF]
    test = df[df[DATETIME_COL] >= TIME_SPLIT_CUTOFF]

    print(f"Train: {len(train)} rows (< {TIME_SPLIT_CUTOFF})")
    print(f"Test:  {len(test)} rows (>= {TIME_SPLIT_CUTOFF})")

    pipeline = FeaturePipeline(
        lookup_path=FEATURE_LOOKUP_PATH,
        encoding_smoothing=FEATURE_ENCODING_SMOOTHING,
        frequency_fill=FEATURE_FREQUENCY_FILL,
        rush_hour_morning_start=FEATURE_RUSH_HOUR_MORNING_START,
        rush_hour_morning_end=FEATURE_RUSH_HOUR_MORNING_END,
        rush_hour_evening_start=FEATURE_RUSH_HOUR_EVENING_START,
        rush_hour_evening_end=FEATURE_RUSH_HOUR_EVENING_END,
        night_start=FEATURE_NIGHT_START,
        night_end=FEATURE_NIGHT_END,
        passenger_count_min=FEATURE_PASSENGER_COUNT_MIN,
        passenger_count_max=FEATURE_PASSENGER_COUNT_MAX,
    )
    pipeline.fit(train)
    y_train = train[TARGET].values if TARGET in train else None
    y_test = test[TARGET].values if TARGET in test else None
    train_feat = pipeline.transform(train)
    test_feat = pipeline.transform(test)

    return train_feat, test_feat, y_train, y_test


def train_lgbm(train_feat: pd.DataFrame,
               y_train: np.ndarray) -> lgb.LGBMRegressor:
    X_train = train_feat[ENGINEERED_FEATURES]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: lgb.LGBMRegressor,
             test_feat: pd.DataFrame,
             y_test: np.ndarray) -> dict:
    X_test = test_feat[ENGINEERED_FEATURES]
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    tail_mask = y_test >= np.quantile(y_test, QUANTILE_TAIL)
    tail_mae = mean_absolute_error(y_test[tail_mask], preds[tail_mask])

    return {"mae": mae, "rmse": rmse, "tail_mae_p90": tail_mae}


def _print_feature_importance(model: lgb.LGBMRegressor) -> None:
    print("\n=== Feature importance (top 15) ===")
    importance = pd.Series(
        model.feature_importances_, index=ENGINEERED_FEATURES
    ).sort_values(ascending=False)
    print(importance.head(15))


def main() -> None:
    print("Loading data with time-based split...")
    train_feat, test_feat, y_train, y_test = load_and_split()

    print("\nTraining LightGBM baseline...")
    model = train_lgbm(train_feat, y_train)

    print("\n=== Holdout performance (time-based split) ===")
    metrics = evaluate(model, test_feat, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    _print_feature_importance(model)

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
