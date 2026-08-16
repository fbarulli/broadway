"""MLflow PyFunc wrapper — loads a pickled model for inference."""

from __future__ import annotations

import pickle
from typing import Any

import mlflow.pyfunc
import pandas as pd


class ModelPyFunc(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined]
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:  # type: ignore[name-defined]
        with open(context.artifacts["model"], "rb") as f:
            self._model = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # type: ignore[name-defined]
        model_input: pd.DataFrame,
        params: dict[str, float | int | str] | None = None,
    ) -> Any:
        return self._model.predict(model_input)
