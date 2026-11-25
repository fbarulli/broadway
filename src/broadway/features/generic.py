from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors

from broadway.config.schema import DatasetContract, PipelineConfig
from broadway.contracts.pandera import pandera_dtype
from broadway.features.plan import ColumnOp, ColumnPlan
from broadway.features.schema import FeatureSpec
from broadway.schemas.joined import joined_schema_columns


def build_generic_feature_specs(
    contract: DatasetContract, plan: ColumnPlan
) -> dict[str, FeatureSpec]:
    """Feature specs in the plan's declared production order.

    Encoding emission order mirrors ``FeaturePipeline.transform``'s write order
    (all frequency encoders before all target encoders), so the ordered
    read-side schema can match a conformant features-step frame.
    """
    return {op.name: _spec_for(contract, op) for op in plan.columns}


def _spec_for(contract: DatasetContract, op: ColumnOp) -> FeatureSpec:
    """One spec, with nullability taken from the contract for source columns."""
    column = contract.columns.get(op.name)
    nullable = op.op == "source" and column is not None and column.null_count > 0
    return FeatureSpec(name=op.name, dtype=op.dtype, nullable=nullable)


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
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    specs = build_generic_feature_specs(cfg.dataset, plan)
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
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    declared = set(cfg.dataset.columns) | set(joined_schema_columns(cfg.dataset)) | set(plan.names)
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
