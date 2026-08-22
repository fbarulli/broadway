"""Build sklearn Pipelines from experiment config preprocessing steps.

Each ``PreprocessingStepConfig`` maps to one transformer (or ``passthrough``)
in config order; an absent ``preprocessing`` block yields an identity
passthrough Pipeline (behavior identical to a bare model). Step columns are
name-driven and enforced against the schema module named by
``data_source.schema_contract`` at config-load (Decision 6).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from broadway.config.schema import PipelineConfig, PreprocessingStepConfig
from broadway.features.transformers import FrequencyEncoding, TargetEncoding
from broadway.schemas import schema_columns

_STEP_ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "target_encoding": frozenset({"smoothing"}),
    "frequency_encoding": frozenset({"normalize"}),
    "one_hot": frozenset(),
    "passthrough": frozenset(),
}

_REQUIRED_STEP_PARAMS: dict[str, frozenset[str]] = {
    "target_encoding": frozenset({"smoothing"}),
    "frequency_encoding": frozenset(),
    "one_hot": frozenset(),
    "passthrough": frozenset(),
}


def _validate_step_params(step: PreprocessingStepConfig) -> None:
    allowed = _STEP_ALLOWED_PARAMS[step.type]
    invalid = sorted(set(step.params) - allowed)
    if invalid:
        raise ValueError(
            f"invalid params for preprocessing step '{step.type}': {invalid}. "
            f"valid params: {sorted(allowed)}"
        )
    missing = sorted(_REQUIRED_STEP_PARAMS[step.type] - set(step.params))
    if missing:
        raise ValueError(
            f"preprocessing step '{step.type}' requires param(s) {missing}"
        )


def _build_step(step: PreprocessingStepConfig, target: str) -> object:
    _validate_step_params(step)
    if step.type == "target_encoding":
        return TargetEncoding(
            columns=step.columns,
            target=target,
            smoothing=float(step.params["smoothing"]),
        )
    if step.type == "frequency_encoding":
        if "normalize" in step.params:
            return FrequencyEncoding(
                columns=step.columns, normalize=bool(step.params["normalize"])
            )
        return FrequencyEncoding(columns=step.columns)
    if step.type == "one_hot":
        return ColumnTransformer(
            [("one_hot", OneHotEncoder(handle_unknown="ignore"), step.columns)],
            remainder="passthrough",
        )
    if step.type == "passthrough":
        return "passthrough"
    raise ValueError(f"unknown preprocessing step type '{step.type}'")


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
            (f"{step.type}_{i}", _build_step(step, cfg.dataset.target))
            for i, step in enumerate(cfg.experiment.preprocessing)
        ]
    )


def validate_preprocessing_columns(cfg: PipelineConfig) -> None:
    """Cross-check recipe columns against the bound schema module (Decision 6).

    Strict subset: every preprocessing step's columns must exist in the
    schema module named by ``data_source.schema_contract``; a violation names
    the offending step and columns and fails config-load.
    """
    if cfg.experiment is None or not cfg.experiment.preprocessing:
        return
    if cfg.dataset is None:
        raise ValueError(
            "cannot cross-check preprocessing columns: experiment has a recipe but no dataset"
        )
    bound = schema_columns(cfg.experiment.data_source.schema_contract, cfg.dataset)
    for step in cfg.experiment.preprocessing:
        missing = sorted(c for c in step.columns if c not in bound)
        if missing:
            raise ValueError(
                f"preprocessing step '{step.type}' references column(s) {missing} not in "
                f"schema_contract '{cfg.experiment.data_source.schema_contract}' "
                f"(available: {sorted(bound)})"
            )
