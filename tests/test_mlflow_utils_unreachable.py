"""Unreachable MLflow HTTP store surfaces one actionable error, no raw spam."""

from __future__ import annotations

import mlflow
import pytest

from broadway.training.mlflow_utils import setup_mlflow

_UNREACHABLE_URI = "http://localhost:5000"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ConnectionError("connection refused"), id="raw-connection-error"),
        pytest.param(
            mlflow.exceptions.MlflowException(
                "API request to http://localhost:5000/api/2.0/mlflow/experiments/get-by-name "
                "failed with exception HTTPConnectionPool(host='localhost', port=5000): "
                "Max retries exceeded with url: /api/2.0/mlflow/experiments/get-by-name "
                "(Caused by NewConnectionError(...): Failed to establish a new connection: "
                "[Errno 111] Connection refused"
            ),
            id="wrapped-mlflow-exception",
        ),
    ],
)
def test_unreachable_http_store_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A refused connection to an http(s) store raises one actionable error."""

    def _refuse(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(mlflow, "set_experiment", _refuse)
    with pytest.raises(RuntimeError) as excinfo:
        setup_mlflow(_UNREACHABLE_URI, "test_experiment")
    message = str(excinfo.value)
    assert "MLflow server unreachable at http://localhost:5000" in message
    assert "uv run mlflow server" in message
    assert excinfo.value.__cause__ is failure
