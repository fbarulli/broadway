from __future__ import annotations

import joblib
import mlflow.pyfunc
import pandas as pd


class TaxiDurationPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    Bundles a fitted FeaturePipeline with a trained model so that
    raw trip data goes in and a duration prediction comes out.

    Logged via MLflow's "models from code" pattern: this file itself
    is logged (not a pickled live object). load_context() restores
    the fitted pipeline/model from a joblib artifact at load time,
    avoiding CloudPickle-serializing an in-memory object, which
    MLflow warns can execute arbitrary code on deserialization.
    """

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["model_bundle"])
        self.feature_pipeline = bundle["feature_pipeline"]
        self.model = bundle["model"]

    def predict(self, context, model_input: pd.DataFrame, params=None):
        features = self.feature_pipeline.transform(model_input)
        return self.model.predict(features)


mlflow.models.set_model(TaxiDurationPyfuncModel())
