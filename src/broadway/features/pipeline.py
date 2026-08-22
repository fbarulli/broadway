"""FeaturePipeline dataclass — fit on train, transform on any DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from broadway.config.schema import EncodingConfig, FeatureConfig
from broadway.features.builders import build_derived, load_custom_builders
from broadway.features.transformers import FrequencyEncoding, TargetEncoding


@dataclass
class FeaturePipeline:
    encodings: list[EncodingConfig] = field(default_factory=list)
    _target_encoders: list[TargetEncoding] = field(default_factory=list)
    _freq_encoders: list[FrequencyEncoding] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, target: str, smoothing: int) -> FeaturePipeline:
        self._target_encoders = []
        self._freq_encoders = []
        for enc in self.encodings:
            for col in enc.columns:
                if enc.type == "target":
                    encoder = TargetEncoding(columns=[col], smoothing=smoothing)
                    self._target_encoders.append(encoder.fit(df, df[target]))
                elif enc.type == "frequency":
                    encoder = FrequencyEncoding(columns=[col])
                    self._freq_encoders.append(encoder.fit(df))
        return self

    def transform(self, df: pd.DataFrame, cfg: FeatureConfig, target: str, freq_fill: float) -> pd.DataFrame:
        result = df.copy()
        result = build_derived(result, cfg.derived, target, extra_builders=load_custom_builders(cfg.builder_module))
        for encoder in self._freq_encoders:
            result = encoder.transform(result, fill=freq_fill)
        for encoder in self._target_encoders:
            result = encoder.transform(result)
        return result
