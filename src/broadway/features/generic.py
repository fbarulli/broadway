from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors

from broadway.config.schema import DatasetContract, FeatureConfig, PipelineConfig
from broadway.contracts.pandera import pandera_dtype
from broadway.features.builders import builder_dtype
from broadway.features.schema import FeatureSpec
from broadway.schemas.joined import joined_schema_columns


def build_generic_feature_specs(contract: DatasetContract, features: FeatureConfig) -> dict[str, FeatureSpec]:
    specs: dict[str, FeatureSpec] = {}
    for col in features.include:
        if col in contract.columns:
            col_schema = contract.columns[col]
            specs[col] = FeatureSpec(name=col, dtype=col_schema.dtype, nullable=col_schema.null_count > 0)
    for derived in features.derived:
        specs[derived.name] = FeatureSpec(name=derived.name, dtype=builder_dtype(derived.func))
    # Encoding emission order mirrors FeaturePipeline.transform's write order:
    # all frequency encoders run before all target encoders, so the ordered
    # read-side schema can match a conformant features-step frame.
    for enc in features.encodings:
        for col in enc.columns:
            if enc.type == "frequency":
                specs[f"{col}_freq_enc"] = FeatureSpec(name=f"{col}_freq_enc", dtype="float64")
    for enc in features.encodings:
        for col in enc.columns:
            if enc.type == "target":
                specs[f"{col}_target_enc"] = FeatureSpec(name=f"{col}_target_enc", dtype="float64")
    return specs


def engineered_schema_for(cfg: PipelineConfig) -> pa.DataFrameSchema:
    """Engineered-feature contract for the model-surface feature files.

    Ordered per the specs SSOT (include, then derived, then frequency- before
    target-encoded columns — the order the features step writes) and
    dtype-checked per the declared dtypes (builder-derived dtypes come from
    ``_BUILDER_DTYPES``). Non-strict: columns outside the declared surface (the
    target, non-eligible columns) are tolerated, so a column-order or dtype
    drift in the persisted file fails loud here without rejecting extra
    columns. Model-surface consumers validate on read so a regression at write
    cannot silently reach training/evaluation.
    """
    assert cfg.dataset is not None and cfg.experiment is not None
    specs = build_generic_feature_specs(cfg.dataset, cfg.experiment.features)
    return pa.DataFrameSchema(
        {
            spec.name: pa.Column(pandera_dtype(spec.dtype), nullable=spec.nullable)
            for spec in specs.values()
        },
        ordered=True,
    )


def validate_target_dtype(cfg: PipelineConfig, df: pd.DataFrame) -> None:
    """Explicit target-dtype check hook for engineered-frame reads.

    The engineered feature schema covers feature columns only — the target is
    excluded from ``eligible_feature_columns`` — so the target column's
    declared dtype is enforced here at the read boundary.
    """
    assert cfg.dataset is not None
    target = cfg.dataset.target
    if target not in df.columns:
        raise SchemaError(
            schema=engineered_schema_for(cfg),
            data=df,
            message=f"target column '{target}' missing from the engineered feature frame",
            failure_cases=target,
            check="column_in_schema",
        )
    declared = cfg.dataset.columns[target].dtype
    actual = str(df[target].dtype)
    if actual != declared:
        raise SchemaError(
            schema=engineered_schema_for(cfg),
            data=df,
            message=(
                f"target column '{target}' dtype {actual} does not match "
                f"declared dtype {declared}"
            ),
            failure_cases=target,
            check="dtype",
        )


def _assert_no_extra_columns(cfg: PipelineConfig, df: pd.DataFrame) -> None:
    """Reject columns outside the declared engineered surface (FIX_4 G2).

    The engineered feature files legitimately carry the full canonical frame —
    contract columns (including the target and non-eligible columns) plus the
    joined loader's lookup-derived columns — alongside the declared feature
    specs, so blanket ``strict=True`` on ``engineered_schema_for`` would reject
    conformant flows (the mlflow/sample surfaces are separate boundaries). This
    explicit guard instead rejects only columns outside the declared surface:
    contract columns ∪ joined-lookup columns ∪ feature specs.
    """
    assert cfg.dataset is not None and cfg.experiment is not None
    declared = (
        set(cfg.dataset.columns)
        | set(joined_schema_columns(cfg.dataset))
        | set(build_generic_feature_specs(cfg.dataset, cfg.experiment.features))
    )
    extra = sorted(set(df.columns) - declared)
    if extra:
        raise SchemaError(
            schema=engineered_schema_for(cfg),
            data=df,
            message=(
                "engineered feature frame contains unexpected column(s): "
                + ", ".join(extra)
            ),
            failure_cases=extra,
            check="column_subset",
        )


def validate_engineered_frame(cfg: PipelineConfig, df: pd.DataFrame) -> None:
    """Validate a read engineered feature frame against the ordered schema,
    then the extra-column guard, then the explicit target-dtype hook. Any
    mismatch raises SchemaError.

    Pandera raises the aggregate ``SchemaErrors`` for column-order violations;
    the read contract requires ``SchemaError`` for every mismatch class.
    """
    schema = engineered_schema_for(cfg)
    try:
        schema.validate(df)
    except SchemaErrors as exc:
        raise SchemaError(
            schema=schema,
            data=df,
            message=f"engineered feature frame violates the ordered schema: {exc}",
        ) from exc
    _assert_no_extra_columns(cfg, df)
    validate_target_dtype(cfg, df)
