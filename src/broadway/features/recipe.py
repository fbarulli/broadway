"""Build sklearn Pipelines from experiment config preprocessing steps.

Each ``PreprocessingStepConfig`` maps to one transformer (or ``passthrough``)
in config order; an absent ``preprocessing`` block yields an identity
passthrough Pipeline (behavior identical to a bare model). Step construction
and param validation are owned by ``broadway.features.transformers`` — this
module only composes the Pipeline and cross-checks declarations. Step columns
are name-driven and enforced against the schema module named by
``data_source.schema_contract`` at config-load (Decision 6).

Target transforms (``difference``/``boxcox``) are declared separately under
``experiment.target_transforms`` and built as an invertible
``TargetTransformChain``; they are never sklearn Pipeline steps because
differencing changes the row count.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from broadway.config.schema import (
    TARGET_TRANSFORM_STEP_TYPES,
    PipelineConfig,
    PreprocessingStepConfig,
)
from broadway.features.transformers import PreprocessingFactory, TargetTransformChain
from broadway.schemas import schema_columns


def _build_step(step: PreprocessingStepConfig) -> object:
    """Build one X-transformer step through the single preprocessing owner.

    Kept as a module seam: the HPO leakage guard patches it to prove the
    composed Pipeline refits preprocessing on every trial.
    """
    return PreprocessingFactory.build(step)


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    """Build the experiment's sklearn Pipeline; absent preprocessing = passthrough identity."""
    if cfg.experiment is None:
        raise ValueError("recipe requires an experiment config")
    if cfg.dataset is None:
        raise ValueError("recipe requires a dataset config")
    if not cfg.experiment.preprocessing:
        return Pipeline([("passthrough", "passthrough")])
    return Pipeline(
        [
            (f"{step.type}_{i}", _build_step(step))
            for i, step in enumerate(cfg.experiment.preprocessing)
        ]
    )


def build_target_transforms(cfg: PipelineConfig) -> TargetTransformChain:
    """Build the experiment's invertible target-transform chain.

    The chain is fitted on the train target only; ``transform`` returns the
    transformed target (differencing drops its boundary rows) and
    ``inverse_transform`` restores target units for scoring.
    """
    if cfg.experiment is None:
        raise ValueError("recipe requires an experiment config")
    if not cfg.experiment.target_transforms:
        raise ValueError(
            "experiment declares no target_transforms — nothing to build "
            "(difference/boxcox belong under experiment.target_transforms)"
        )
    return TargetTransformChain(
        [
            PreprocessingFactory.build_target_transform(step)
            for step in cfg.experiment.target_transforms
        ]
    )


def validate_preprocessing_columns(cfg: PipelineConfig) -> None:
    """Cross-check recipe columns against the bound schema module (Decision 6).

    Strict subset: every preprocessing step's columns must exist in the
    schema module named by ``data_source.schema_contract``; a violation names
    the offending step and columns and fails config-load. ``one_hot`` must be
    the SOLE preprocessing step: its ColumnTransformer emits an ndarray that
    breaks chained encoders, and its remainder passthrough leaks raw
    categorical columns into the model. Target transforms must name exactly the
    dataset's target column.
    """
    if cfg.experiment is None:
        return
    if cfg.experiment.target_transforms:
        _validate_target_columns(cfg, cfg.experiment.target_transforms)
    if not cfg.experiment.preprocessing:
        return
    if cfg.dataset is None:
        raise ValueError(
            "cannot cross-check preprocessing columns: experiment has a recipe but no dataset"
        )
    steps = cfg.experiment.preprocessing
    if any(step.type == "one_hot" for step in steps) and len(steps) > 1:
        # upgrade path: a frame-output ColumnTransformer (set_output(transform="pandas"))
        # lifts this restriction — until then one_hot stays single-step.
        raise ValueError(
            f"preprocessing recipe {[step.type for step in steps]} mixes one_hot with "
            "other steps: one_hot must be the sole preprocessing step (its "
            "ColumnTransformer emits an ndarray that breaks chained encoders, and "
            "the remainder passthrough leaks raw categorical columns). Use one_hot "
            "alone, or move the other steps into a separate recipe."
        )
    bound = schema_columns(
        cfg.experiment.data_source.schema_contract,
        cfg.dataset,
        plan=cfg.experiment.plan,
    )
    for step in cfg.experiment.preprocessing:
        missing = sorted(c for c in step.columns if c not in bound)
        if missing:
            raise ValueError(
                f"preprocessing step '{step.type}' references column(s) {missing} not in "
                f"schema_contract '{cfg.experiment.data_source.schema_contract}' "
                f"(available: {sorted(bound)})"
            )


def _validate_target_columns(
    cfg: PipelineConfig, steps: list[PreprocessingStepConfig]
) -> None:
    """Every target transform must name exactly the dataset's target column."""
    if cfg.dataset is None:
        raise ValueError("cannot cross-check target_transforms: experiment has no dataset")
    for step in steps:
        if step.type not in TARGET_TRANSFORM_STEP_TYPES:
            raise ValueError(
                f"preprocessing step '{step.type}' is not a target transform "
                f"(valid: {sorted(t.value for t in TARGET_TRANSFORM_STEP_TYPES)})"
            )
        if step.columns != [cfg.dataset.target]:
            raise ValueError(
                f"target transform '{step.type}' must name exactly the dataset target "
                f"'{cfg.dataset.target}', got {step.columns}"
            )
