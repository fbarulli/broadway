from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from broadway.features.basic import add_basic_features
from broadway.features.boroughs import add_borough_features, load_zones
from broadway.features.contracts import validate_engineered_schema
from broadway.features.ml_encodings import apply_target_encoding, make_target_encoding
from broadway.features.frequency import apply_frequency_encoding, make_frequency_encoding
from broadway.features.schema import ENGINEERED_FEATURES, ROUTE_KEYS, TARGET


@dataclass
class FeaturePipeline:
    lookup_path: str = ""
    encoding_smoothing: int = 50
    frequency_fill: float = 0.0
    zones: pd.DataFrame | None = field(default=None, init=False)
    route_stats: pd.DataFrame | None = field(default=None, init=False)
    route_frequency: pd.DataFrame | None = field(default=None, init=False)
    global_mean: float | None = field(default=None, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        self.zones = load_zones(self.lookup_path)

        engineered = self._add_deterministic_features(train_df)

        self.route_stats, self.global_mean = make_target_encoding(
            engineered,
            ROUTE_KEYS,
            TARGET,
            "route_avg_duration",
            self.encoding_smoothing,
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
            self.frequency_fill,
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
