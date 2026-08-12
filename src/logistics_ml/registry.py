import mlflow


def register_model(
    model_name: str,
    run_id: str,
):
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(
        f"Registered model: {result.name} "
        f"version {result.version}"
    )

    return result
