"""LGBM non-linear baseline — routed here from 08-11: OLS assumptions failed
under all remediations (kurtosis=25.5, Breusch-Pagan/Jarque-Bera p≈0)."""

import numpy as np
import pandas as pd

from projects.taxi import data
from broadway.features.ml_pipeline import FeaturePipeline
from broadway.features.schema import TARGET
from projects.taxi.features import ENGINEERED_FEATURES
from broadway.stats import baseline


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(data.DATA_PATH)
    if len(df) > data.SAMPLE_SIZE:
        df = df.sample(n=data.SAMPLE_SIZE, random_state=data.RANDOM_STATE)
    df[data.DATETIME_COL] = pd.to_datetime(df[data.DATETIME_COL])
    df = df.sort_values(data.DATETIME_COL)

    train = df[df[data.DATETIME_COL] < data.TIME_SPLIT_CUTOFF]
    test = df[df[data.DATETIME_COL] >= data.TIME_SPLIT_CUTOFF]

    print(f"Train: {len(train)} rows (< {data.TIME_SPLIT_CUTOFF})")
    print(f"Test:  {len(test)} rows (>= {data.TIME_SPLIT_CUTOFF})")

    pipeline = FeaturePipeline(
        lookup_path=data.FEATURE_LOOKUP_PATH,
        encoding_smoothing=data.FEATURE_ENCODING_SMOOTHING,
        frequency_fill=data.FEATURE_FREQUENCY_FILL,
        rush_hour_morning_start=data.FEATURE_RUSH_HOUR_MORNING_START,
        rush_hour_morning_end=data.FEATURE_RUSH_HOUR_MORNING_END,
        rush_hour_evening_start=data.FEATURE_RUSH_HOUR_EVENING_START,
        rush_hour_evening_end=data.FEATURE_RUSH_HOUR_EVENING_END,
        night_start=data.FEATURE_NIGHT_START,
        night_end=data.FEATURE_NIGHT_END,
        passenger_count_min=data.FEATURE_PASSENGER_COUNT_MIN,
        passenger_count_max=data.FEATURE_PASSENGER_COUNT_MAX,
    )
    pipeline.fit(train)
    y_train = train[TARGET].values if TARGET in train else None
    y_test = test[TARGET].values if TARGET in test else None
    train_feat = pipeline.transform(train)
    test_feat = pipeline.transform(test)

    return train_feat, test_feat, y_train, y_test


def _print_feature_importance(model: object) -> None:
    print("\n=== Feature importance (top 15) ===")
    importance = pd.Series(
        model.feature_importances_, index=ENGINEERED_FEATURES
    ).sort_values(ascending=False)
    print(importance.head(15))


def main() -> None:
    print("Loading data with time-based split...")
    train_feat, test_feat, y_train, y_test = load_and_split()

    print("\nTraining LightGBM baseline...")
    model = baseline.train_lgbm(
        train_feat[ENGINEERED_FEATURES],
        y_train,
        objective="regression",
        n_estimators=data.N_ESTIMATORS,
        learning_rate=data.LEARNING_RATE,
        num_leaves=data.NUM_LEAVES,
        subsample=data.SUBSAMPLE,
        colsample_bytree=data.COLSAMPLE_BYTREE,
        random_state=data.RANDOM_STATE,
    )

    print("\n=== Holdout performance (time-based split) ===")
    metrics = baseline.evaluate(
        model,
        test_feat[ENGINEERED_FEATURES],
        y_test,
        tail_quantile=data.QUANTILE_TAIL,
    )
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
