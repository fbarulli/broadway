"""MLflow PyFunc wrapper — loads a pickled bare model for inference.

SCOPE: kept ONLY for previously logged bare-model artifacts (backward
compat). New-path artifacts are sklearn Pipelines logged with an explicit
signature; they load through MLflow's native pyfunc flavor, which carries
preprocessing inside the Pipeline and enforces the logged signature at
predict time. This wrapper is a plain pickled-model loader — no functional
change."""

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
