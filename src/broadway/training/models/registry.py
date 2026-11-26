"""Model registry — the single canonical source of model identity.

The registry key is the ONLY canonical model name: factories live under
`_REGISTRY`, and `MODEL_META` carries each key's display name, allowed
params, and default params. Configs and experiments name models by key;
historical display aliases (e.g. "ols") are presentation-only.
"""

from __future__ import annotations

from typing import Any

from broadway.training.models.lightgbm import create as create_lgbm
from broadway.training.models.lightgbm import create_quantile as create_lgbm_quantile
from broadway.training.models.linear import create as create_linear
from broadway.training.models.random_forest import create as create_rf
from broadway.training.models.xgboost import create as create_xgb

_REGISTRY = {
    "linear": create_linear,
    "lgbm": create_lgbm,
    "lgbm_quantile": create_lgbm_quantile,
    "rf": create_rf,
    "xgb": create_xgb,
}

# LightGBM's native params. ``objective``/``alpha`` let HPO tune a single
# quantile model (``objective="quantile"``); the ``lgbm_quantile`` key below
# owns its objective internally and reads ``alpha`` as the interval miscoverage.
_LGBM_PARAMS = frozenset({
    "n_estimators", "learning_rate", "num_leaves", "max_depth",
    "subsample", "colsample_bytree", "min_child_samples",
    "reg_alpha", "reg_lambda", "random_state", "n_jobs",
    "objective", "alpha",
})

MODEL_META: dict[str, dict] = {
    "linear": {
        "display": "ols",
        "allowed_params": frozenset({"fit_intercept", "positive", "copy_X", "n_jobs"}),
        "default_params": {},
    },
    "lgbm": {
        "display": "lgbm",
        "allowed_params": _LGBM_PARAMS,
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "verbosity": -1,
        },
    },
    "lgbm_quantile": {
        "display": "lgbm_quantile",
        # The bundle owns objective="quantile"; ``alpha`` is the interval
        # miscoverage (0.05 -> 2.5/50/97.5% quantiles), so a caller cannot
        # silently reassign either.
        "allowed_params": _LGBM_PARAMS - {"objective"},
        "default_params": {
            "alpha": 0.05,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "verbosity": -1,
        },
    },
    "xgb": {
        "display": "xgb",
        "allowed_params": frozenset({
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "reg_alpha", "gamma",
            "random_state", "n_jobs", "tree_method",
        }),
        "default_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "verbosity": 0,
            "tree_method": "hist",
        },
    },
    "rf": {
        "display": "rf",
        "allowed_params": frozenset({
            "n_estimators", "max_depth", "max_samples", "min_samples_split",
            "min_samples_leaf", "random_state", "n_jobs",
        }),
        "default_params": {},
    },
}


def get_model(name: str, **params: float | str) -> Any:
    """Build a model: registry defaults merged under explicit params."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model: {name}. valid: {list(_REGISTRY)}")
    merged = dict(MODEL_META[name]["default_params"])
    merged.update(params)
    return _REGISTRY[name](**merged)


def display_name(key: str) -> str:
    """Human-facing display name for a registry key."""
    if key not in MODEL_META:
        raise KeyError(f"unknown model key: {key}. valid: {list(MODEL_META)}")
    return str(MODEL_META[key]["display"])


def allowed_params(key: str) -> frozenset[str]:
    """Params a search space may tune for a key (unknown key -> empty)."""
    meta = MODEL_META.get(key)
    if meta is None:
        return frozenset()
    return meta["allowed_params"]


def model_keys() -> list[str]:
    """All canonical registry keys, in registry order."""
    return list(_REGISTRY)
