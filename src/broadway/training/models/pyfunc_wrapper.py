"""MLflow PyFunc wrapper — loads a pickled model for inference."""

from __future__ import annotations

import pickle
from typing import Any

import mlflow.pyfunc
import pandas as pd


class ModelPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with open(context.artifacts["model"], "rb") as f:
            self._model = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, float | int | str] | None = None,
    ) -> Any:
        return self._model.predict(model_input)
