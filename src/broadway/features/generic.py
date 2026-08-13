from __future__ import annotations

from broadway.config.schema import DatasetContract, FeatureConfig
from broadway.features.builders import builder_dtype
from broadway.features.schema import FeatureSpec


def build_generic_feature_specs(contract: DatasetContract, features: FeatureConfig) -> dict[str, FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    for col in features.include:
        if col in contract.columns:
            col_schema = contract.columns[col]
            specs[col] = FeatureSpec(name=col, dtype=col_schema.dtype, nullable=col_schema.null_count > 0)
    for derived in features.derived:
        specs[derived.name] = FeatureSpec(name=derived.name, dtype=builder_dtype(derived.func))
    for enc in features.encodings:
        for col in enc.columns:
            if enc.type == "target":
                specs[f"{col}_target_enc"] = FeatureSpec(name=f"{col}_target_enc", dtype="float64")
            elif enc.type == "frequency":
                specs[f"{col}_freq_enc"] = FeatureSpec(name=f"{col}_freq_enc", dtype="float64")
    return specs
