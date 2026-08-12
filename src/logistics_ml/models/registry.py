from .lightgbm import LightGBMModel
from .linear import LinearModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel


MODELS = {
    "linear": LinearModel,
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
    "lgbm": LightGBMModel,
}


def get_model(name: str, **params):

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'")

    return MODELS[name](**params)
