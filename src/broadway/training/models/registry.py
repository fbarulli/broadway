"""Model registry — returns model instance by name."""

from __future__ import annotations

from typing import Any

from broadway.training.models.lightgbm import create as create_lgbm
from broadway.training.models.linear import create as create_linear
from broadway.training.models.random_forest import create as create_rf
from broadway.training.models.xgboost import create as create_xgb

_REGISTRY = {
    "linear": create_linear,
    "lgbm": create_lgbm,
    "rf": create_rf,
    "xgb": create_xgb,
}


def get_model(name: str, **params: float | str) -> Any:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model: {name}. valid: {list(_REGISTRY)}")
    return _REGISTRY[name](**params)
