"""Model registry — returns model instance by name."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from broadway.training.models.linear import create as create_linear

_REGISTRY = {
    "linear": create_linear,
}


def get_model(name: str, **params: float | int | str) -> LinearRegression:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model: {name}. valid: {list(_REGISTRY)}")
    return _REGISTRY[name](**params)
