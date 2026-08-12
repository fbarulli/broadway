from __future__ import annotations

import pandas as pd

from broadway.features.ml_encodings import apply_target_encoding, make_target_encoding
from broadway.features.frequency import apply_frequency_encoding, make_frequency_encoding
from broadway.features.schema import ROUTE_KEYS, TARGET
from project.basic import add_basic_features
from project.boroughs import add_borough_features, load_zones
from project.features import ENGINEERED_FEATURES, ENGINEERED_SCHEMA


class FeaturePipeline:
    def __init__(
        self,
        lookup_path: str,
        encoding_smoothing: int,
        frequency_fill: float,
        rush_hour_morning_start: int,
        rush_hour_morning_end: int,
        rush_hour_evening_start: int,
        rush_hour_evening_end: int,
        night_start: int,
        night_end: int,
        passenger_count_min: int,
        passenger_count_max: int,
    ) -> None:
        self.lookup_path = lookup_path
        self.encoding_smoothing = encoding_smoothing
        self.frequency_fill = frequency_fill
        self.rush_hour_morning_start = rush_hour_morning_start
        self.rush_hour_morning_end = rush_hour_morning_end
        self.rush_hour_evening_start = rush_hour_evening_start
        self.rush_hour_evening_end = rush_hour_evening_end
        self.night_start = night_start
        self.night_end = night_end
        self.passenger_count_min = passenger_count_min
        self.passenger_count_max = passenger_count_max
        self.zones: pd.DataFrame | None = None
        self.route_stats: pd.DataFrame | None = None
        self.route_frequency: pd.DataFrame | None = None
        self.global_mean: float | None = None
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
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

    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = apply_target_encoding(
            df, self.route_stats, ROUTE_KEYS, "route_avg_duration", self.global_mean,
        )
        df = apply_frequency_encoding(
            df, self.route_frequency, ROUTE_KEYS, "route_frequency", self.frequency_fill,
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fit() before transform().")
        input_rows = len(df)
        engineered = self._add_deterministic_features(df)
        engineered = self._apply_encodings(engineered)
        if len(engineered) != input_rows:
            raise RuntimeError(
                "FeaturePipeline.transform() changed row count from "
                f"{input_rows} to {len(engineered)}. This usually means a "
                "merge inside the pipeline matched a key more than once "
                "(duplicate rows in zones, route_stats, or route_frequency). "
                "Row-to-row alignment with the input dataframe can no "
                "longer be trusted."
            )
        result = engineered[list(ENGINEERED_FEATURES)]
        return ENGINEERED_SCHEMA.validate(result)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def _add_deterministic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_basic_features(
            df,
            rush_hour_morning_start=self.rush_hour_morning_start,
            rush_hour_morning_end=self.rush_hour_morning_end,
            rush_hour_evening_start=self.rush_hour_evening_start,
            rush_hour_evening_end=self.rush_hour_evening_end,
            night_start=self.night_start,
            night_end=self.night_end,
            passenger_count_min=self.passenger_count_min,
            passenger_count_max=self.passenger_count_max,
        )
        df = add_borough_features(df, self.zones)
        return df
