"""FeaturePipeline dataclass — fit on train, transform on any DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from broadway.config.schema import EncodingConfig, FeatureConfig
from broadway.features.builders import build_derived
from broadway.features.encodings import (
    fit_frequency_encoding,
    fit_target_encoding,
    transform_frequency_encoding,
    transform_target_encoding,
)


@dataclass
class FeaturePipeline:
    encodings: list[EncodingConfig] = field(default_factory=list)
    _target_mappings: dict[str, dict[str, float]] = field(default_factory=dict)
    _freq_mappings: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, target: str, smoothing: int) -> FeaturePipeline:
        for enc in self.encodings:
            for col in enc.on:
                if enc.type == "target":
                    self._target_mappings[col] = fit_target_encoding(df, col, target, smoothing)
                elif enc.type == "frequency":
                    self._freq_mappings[col] = fit_frequency_encoding(df, col)
        return self

    def transform(self, df: pd.DataFrame, cfg: FeatureConfig, target: str, freq_fill: float) -> pd.DataFrame:
        result = df.copy()
        result = build_derived(result, cfg.derived, target)
        for col, mapping in self._freq_mappings.items():
            result = transform_frequency_encoding(result, col, mapping, freq_fill)
        for col, mapping in self._target_mappings.items():
            result = transform_target_encoding(result, col, mapping)
        return result
