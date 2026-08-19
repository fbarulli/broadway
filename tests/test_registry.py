"""Model registry tests — canonical keys, defaults, display names, validation.

Synthetic only: constructed estimators are never fitted, so no data is needed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from broadway.config.schema import (
    ExperimentConfig,
    FeatureConfig,
    HPOConfig,
    ModelConfig,
    ModelHPOSpec,
    SplitConfig,
)
from broadway.training.models.registry import (
    MODEL_META,
    allowed_params,
    display_name,
    get_model,
    model_keys,
)


def test_get_model_applies_registry_defaults() -> None:
    model = get_model("lgbm")
    assert model.n_estimators == 100
    assert model.learning_rate == 0.1
    assert model.max_depth == 5
    assert model.verbosity == -1


def test_get_model_explicit_params_override_defaults() -> None:
    model = get_model("xgb", n_estimators=250)
    assert model.n_estimators == 250
    assert model.tree_method == "hist"


def test_get_model_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_model("not_a_model")


def test_display_name_linear_is_ols() -> None:
    assert display_name("linear") == "ols"
    assert display_name("lgbm") == "lgbm"


def test_model_keys_match_meta_keys() -> None:
    assert set(model_keys()) == {"linear", "lgbm", "rf", "xgb"}
    assert set(model_keys()) == set(MODEL_META)


def test_allowed_params_match_schema_validation() -> None:
    assert "fit_intercept" in allowed_params("linear")
    assert "max_depth" not in allowed_params("linear")
    assert {"n_estimators", "learning_rate", "num_leaves"} <= allowed_params("lgbm")


def test_schema_rejects_search_space_outside_allowed_params() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            features=FeatureConfig(include=["rooms"], exclude=[], derived=[], encodings=[]),
            model=ModelConfig(type="lgbm", params={}),
            split=SplitConfig(type="random", validation_size=0.2),
            random_state=42,
            target_metric="rmse",
            hpo=HPOConfig(
                engine="optuna",
                total_trials=10,
                initial_trials_per_model=5,
                top_k=1,
                target_metric="rmse",
                models=[ModelHPOSpec(name="lgbm", search_space={"reg_lambda": [0.1, 1.0]})],
            ),
        )


def test_allowed_params_unknown_key_empty() -> None:
    # Schema treats an unknown model's search space as invalid (empty allowlist).
    assert allowed_params("ghost") == frozenset()
