from __future__ import annotations

import pandas as pd

from broadway.features.transformers import FrequencyEncoding, TargetEncoding
from project.basic import add_basic_features
from project.boroughs import add_borough_features, load_zones
from project.features import ENGINEERED_FEATURES, ENGINEERED_SCHEMA, ROUTE_KEYS, TARGET


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
        self.zones: pd.DataFrame | None = None
        self._route_target_encoder: TargetEncoding | None = None
        self._route_frequency_encoder: FrequencyEncoding | None = None
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        self.zones = load_zones(self.lookup_path)

        engineered = self._add_deterministic_features(train_df)

        self._route_target_encoder = TargetEncoding(
            columns=ROUTE_KEYS,
            target=TARGET,
            smoothing=self.encoding_smoothing,
            feature_name="route_avg_duration",
        ).fit(engineered)

        self._route_frequency_encoder = FrequencyEncoding(
            columns=ROUTE_KEYS,
            feature_name="route_frequency",
            normalize=False,
        ).fit(engineered)

        self.fitted = True

        return self

    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        target_encoder = self._route_target_encoder
        frequency_encoder = self._route_frequency_encoder
        if target_encoder is None or frequency_encoder is None:
            raise RuntimeError("FeaturePipeline must be fit() before transform().")
        df = target_encoder.transform(df)
        df = frequency_encoder.transform(df, fill=self.frequency_fill)
        df["route_frequency"] = df["route_frequency"].astype("int32")
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
                "(duplicate rows in zones or lookup tables). "
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
        )
        df = add_borough_features(df, self.zones)
        return df
