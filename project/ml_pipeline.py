from __future__ import annotations

import pandas as pd

from broadway.features.generic import build_generic_feature_specs
from broadway.features.schema import build_engineered_schema
from broadway.features.transformers import FrequencyEncoding
from project.basic import add_basic_features
from project.boroughs import add_borough_features, load_zones
from project.features import CONTRACT, PLAN, ROUTE_KEYS

_ENGINEERED_SCHEMA = build_engineered_schema(build_generic_feature_specs(CONTRACT, PLAN))


class FeaturePipeline:
    def __init__(
        self,
        lookup_path: str,
        frequency_fill: float,
        rush_hour_morning_start: int,
        rush_hour_morning_end: int,
        rush_hour_evening_start: int,
        rush_hour_evening_end: int,
        night_start: int,
        night_end: int,
    ) -> None:
        self.lookup_path = lookup_path
        self.frequency_fill = frequency_fill
        self.rush_hour_morning_start = rush_hour_morning_start
        self.rush_hour_morning_end = rush_hour_morning_end
        self.rush_hour_evening_start = rush_hour_evening_start
        self.rush_hour_evening_end = rush_hour_evening_end
        self.night_start = night_start
        self.night_end = night_end
        self.zones: pd.DataFrame | None = None
        self._route_frequency_encoder: FrequencyEncoding | None = None
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        self.zones = load_zones(self.lookup_path)

        engineered = self._add_deterministic_features(train_df)

        self._route_frequency_encoder = FrequencyEncoding(
            columns=list(ROUTE_KEYS),
            feature_name="route_frequency",
            normalize=False,
        ).fit(engineered)

        self.fitted = True

        return self

    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        frequency_encoder = self._route_frequency_encoder
        if frequency_encoder is None:
            raise RuntimeError("FeaturePipeline must be fit() before transform().")
        encoded = frequency_encoder.transform(df, fill=self.frequency_fill)
        if len(encoded) != len(df):
            raise RuntimeError(
                "FeaturePipeline.transform() changed row count from "
                f"{len(df)} to {len(encoded)}. This usually means a "
                "merge inside the pipeline matched a key more than once "
                "(duplicate rows in zones or lookup tables). "
                "Row-to-row alignment with the input dataframe can no "
                "longer be trusted."
            )
        encoded["route_frequency"] = encoded["route_frequency"].astype("int32")
        return encoded

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
        result, _ = PLAN.apply(engineered)
        return _ENGINEERED_SCHEMA.validate(result)

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
