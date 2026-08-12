from __future__ import annotations

import pandas as pd

from broadway.features.ml_pipeline import FeaturePipeline
from broadway.features.schema import TARGET


def prepare_dataset(df: pd.DataFrame, train_test_cutoff: str) -> tuple:
    cutoff = pd.Timestamp(train_test_cutoff)

    train_df = df[df["pickup_datetime"] < cutoff].copy().reset_index(drop=True)
    test_df = df[df["pickup_datetime"] >= cutoff].copy().reset_index(drop=True)

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    train_features = pipeline.transform(train_df)
    test_features = pipeline.transform(test_df)

    X_train = train_features
    y_train = train_df.loc[train_features.index, TARGET]

    X_test = test_features
    y_test = test_df.loc[test_features.index, TARGET]

    return X_train, X_test, y_train, y_test, pipeline
