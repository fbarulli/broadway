from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from logistics_ml.config import data as data_config

from .basic import add_basic_features
from .boroughs import add_borough_features, load_zones
from .contracts import validate_engineered_schema
from .encodings import apply_target_encoding, make_target_encoding
from .frequency import apply_frequency_encoding, make_frequency_encoding
from .schema import ENGINEERED_FEATURES, ROUTE_KEYS, TARGET


@dataclass
class FeaturePipeline:
    """
    Learns and applies feature engineering.

    fit() learns statistics from training data.
    transform() applies the learned transformations.
    """

    zones: pd.DataFrame | None = None
    route_stats: pd.DataFrame | None = None
    route_frequency: pd.DataFrame | None = None
    global_mean: float | None = None
    fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        self.zones = load_zones(data_config.taxi_lookup)

        engineered = self._add_deterministic_features(train_df)

        self.route_stats, self.global_mean = make_target_encoding(
            engineered,
            ROUTE_KEYS,
            TARGET,
            "route_avg_duration",
            data_config.encoding_smoothing,
        )

        self.route_frequency = make_frequency_encoding(
            engineered,
            ROUTE_KEYS,
            "route_frequency",
        )

        self.fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError(
                "FeaturePipeline must be fit() before transform()."
            )

        input_rows = len(df)

        engineered = self._add_deterministic_features(df)

        engineered = apply_target_encoding(
            engineered,
            self.route_stats,
            ROUTE_KEYS,
            "route_avg_duration",
            self.global_mean,
        )

        engineered = apply_frequency_encoding(
            engineered,
            self.route_frequency,
            ROUTE_KEYS,
            "route_frequency",
            data_config.frequency_fill,
        )

        if len(engineered) != input_rows:
            raise RuntimeError(
                "FeaturePipeline.transform() changed row count from "
                f"{input_rows} to {len(engineered)}. This usually means a "
                "merge inside the pipeline matched a key more than once "
                "(duplicate rows in zones, route_stats, or route_frequency). "
                "Row-to-row alignment with the input dataframe can no "
                "longer be trusted."
            )

        result = engineered[ENGINEERED_FEATURES]
        validate_engineered_schema(result)

        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def _add_deterministic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_basic_features(df)
        df = add_borough_features(df, self.zones)
        return df
