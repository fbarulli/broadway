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

# one_hot and passthrough take ZERO recipe params BY DESIGN: _build_step
# hardcodes OneHotEncoder(handle_unknown="ignore") internally (single-use, no
# config surface) and passthrough is the identity step — the empty tuples are
# intentional, not unimplemented.
_STEP_PARAM_SPEC: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    #                       required                    optional
    "target_encoding":    (frozenset({"smoothing"}),   frozenset()),
    "frequency_encoding": (frozenset(),                frozenset({"normalize"})),
    "one_hot":            (frozenset(),                frozenset()),
    "passthrough":        (frozenset(),                frozenset()),
}


def _validate_step_params(step: PreprocessingStepConfig) -> None:
    required, optional = _STEP_PARAM_SPEC[step.type]
    allowed = required | optional
    invalid = sorted(set(step.params) - allowed)
    if invalid:
        raise ValueError(
            f"invalid params for preprocessing step '{step.type}': {invalid}. "
            f"valid params: {sorted(allowed)}"
        )
    missing = sorted(required - set(step.params))
    if missing:
        raise ValueError(
            f"preprocessing step '{step.type}' requires param(s) {missing}"
        )


def _coerce_bool_param(value: object, step: str, param: str) -> bool:
    """Accept real bools and case-insensitive 'true'/'false'; fail loud otherwise."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(
        f"preprocessing step '{step}' param '{param}' must be a bool or "
        f"'true'/'false' (case-insensitive), got {value!r}"
    )


def _build_step(step: PreprocessingStepConfig) -> object:
    _validate_step_params(step)
    if step.type == "target_encoding":
        # literal param name is safe: _validate_step_params guarantees presence
        return TargetEncoding(
            columns=step.columns,
            smoothing=float(step.params["smoothing"]),
        )
    if step.type == "frequency_encoding":
        if "normalize" in step.params:
            return FrequencyEncoding(
                columns=step.columns,
                normalize=_coerce_bool_param(step.params["normalize"], step.type, "normalize"),
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
            (f"{step.type}_{i}", _build_step(step))
            for i, step in enumerate(cfg.experiment.preprocessing)
        ]
    )


def validate_preprocessing_columns(cfg: PipelineConfig) -> None:
    """Cross-check recipe columns against the bound schema module (Decision 6).

    Strict subset: every preprocessing step's columns must exist in the
    schema module named by ``data_source.schema_contract``; a violation names
    the offending step and columns and fails config-load. ``one_hot`` must be
    the SOLE preprocessing step: its ColumnTransformer emits an ndarray that
    breaks chained encoders, and its remainder passthrough leaks raw
    categorical columns into the model.
    """
    if cfg.experiment is None or not cfg.experiment.preprocessing:
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
    bound = schema_columns(cfg.experiment.data_source.schema_contract, cfg.dataset)
    for step in cfg.experiment.preprocessing:
        missing = sorted(c for c in step.columns if c not in bound)
        if missing:
            raise ValueError(
                f"preprocessing step '{step.type}' references column(s) {missing} not in "
                f"schema_contract '{cfg.experiment.data_source.schema_contract}' "
                f"(available: {sorted(bound)})"
            )
