import time
import uuid
from logistics_ml.config.mlflow import mlflow as mlflow_config

from fastapi import FastAPI
from pydantic import create_model
import mlflow
import pandas as pd

from logistics_ml.features.schema import RAW_FEATURE_TYPES

app = FastAPI()

MODEL_NAME = mlflow_config.registered_model_name
MODEL_ALIAS = mlflow_config.champion_alias

TaxiTrip = create_model(
    "TaxiTrip",
    **{name: (typ, ...) for name, typ in RAW_FEATURE_TYPES.items()},
)

model = None
prediction_count = 0
total_prediction_time = 0.0
model_load_time = 0.0
model_ready = False


def get_model():
    global model
    global model_load_time
    if model is None:
        start = time.time()
        print("=== LOADING MODEL ===")
        tracking_uri = mlflow_config.tracking_uri
        print("MLFLOW URI:")
        print(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )
        model_uri = f"models:/{MODEL_NAME}/{mv.version}"
        print("RESOLVED MODEL VERSION:")
        print(mv.version)
        print("MODEL URI:")
        print(model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        model_load_time = time.time() - start
        print("MODEL LOADED")
        print(
            f"MODEL LOAD TIME: {model_load_time * 1000:.2f} ms"
        )
    return model


@app.on_event("startup")
def startup():
    global model_ready
    get_model()
    model_ready = True


@app.get("/health")
def health():
    return {
        "status": "ok" if model_ready else "loading"
    }


@app.get("/model")
def model_info():
    tracking_uri = mlflow_config.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )
    return {
        "model": MODEL_NAME,
        "version": mv.version,
        "alias": MODEL_ALIAS,
    }


@app.post("/predict")
def predict(trip: TaxiTrip):
    global prediction_count
    global total_prediction_time

    request_id = str(uuid.uuid4())
    model = get_model()

    data = pd.DataFrame([trip.model_dump()])

    start = time.time()
    prediction = model.predict(data)
    latency = time.time() - start

    prediction_count += 1
    total_prediction_time += latency

    latency_ms = round(latency * 1000, 2)

    print(
        f"request_id={request_id} "
        f"prediction_latency_ms={latency_ms}"
    )

    return {
        "request_id": request_id,
        "prediction": float(prediction[0]),
        "latency_ms": latency_ms,
    }


@app.get("/metrics")
def metrics():
    avg_latency = 0
    if prediction_count:
        avg_latency = (
            total_prediction_time / prediction_count
        ) * 1000
    return {
        "predictions": prediction_count,
        "average_prediction_latency_ms": round(avg_latency, 2),
        "model_load_time_ms": round(model_load_time * 1000, 2),
        "model_ready": model_ready,
    }
