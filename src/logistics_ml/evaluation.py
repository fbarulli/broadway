import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, predictions),
    }


def compare_models(candidate, champion, X_test, y_test):
    return {
        "candidate": evaluate(candidate, X_test, y_test),
        "champion": evaluate(champion, X_test, y_test),
    }


def should_promote(
    candidate_metrics,
    champion_metrics,
    metric="rmse",
):
    if metric == "rmse":
        return candidate_metrics["rmse"] < champion_metrics["rmse"]

    if metric == "mae":
        return candidate_metrics["mae"] < champion_metrics["mae"]

    if metric == "r2":
        return candidate_metrics["r2"] > champion_metrics["r2"]

    raise ValueError(f"Unknown metric '{metric}'")
