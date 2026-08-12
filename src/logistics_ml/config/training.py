from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    train_test_cutoff: str = "2024-01-25"
    random_state: int = 42

    validation_size: float = 0.2

    metric: str = "rmse"

    maximize: bool = False

    default_model: str = "lightgbm"

    supported_models = [
        "lightgbm",
        "xgboost",
    ]


training = TrainingConfig()
