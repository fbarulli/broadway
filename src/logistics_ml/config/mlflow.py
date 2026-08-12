from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MLflowConfig:

    tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    )

    experiment_name: str = "taxi-duration-prediction"

    registered_model_name: str = "taxi-duration-model"

    champion_alias: str = "champion"


mlflow = MLflowConfig()
