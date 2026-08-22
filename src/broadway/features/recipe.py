"""Build sklearn Pipelines from experiment config preprocessing steps."""

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


def _validate_step_params(step: PreprocessingStepConfig) -> None:
    allowed = _STEP_ALLOWED_PARAMS[step.type]
    invalid = sorted(set(step.params) - allowed)
    if invalid:
        raise ValueError(
            f"invalid params for preprocessing step '{step.type}': {invalid}. "
            f"valid params: {sorted(allowed)}"
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
        return FrequencyEncoding(
            columns=step.columns, normalize=bool(step.params.get("normalize", True))
        )
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
    """Cross-check preprocessing columns against the bound schema module (Decision 6)."""
    if cfg.experiment is None or cfg.dataset is None:
        return
    bound = schema_columns(cfg.experiment.data_source.schema_contract, cfg.dataset)
    for step in cfg.experiment.preprocessing:
        missing = sorted(c for c in step.columns if c not in bound)
        if missing:
            raise ValueError(
                f"preprocessing step '{step.type}' references column(s) {missing} not in "
                f"schema_contract '{cfg.experiment.data_source.schema_contract}' "
                f"(columns: {sorted(bound)})"
            )
