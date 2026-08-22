"""Unified HPO API tests — bandit allocation, objective, studies, run_hpo.

Synthetic only: the run_hpo orchestration tests stub make_objective with a
parabola objective so no real model training or external data is involved. The
mlflow tracking tests log runs to a hermetic tmp file store (no server) and
fit a linear model on the tiny 4-row fixture.
"""

from __future__ import annotations

from typing import Self

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureConfig,
    HPOConfig,
    ModelConfig,
    ModelHPOSpec,
    PipelineConfig,
    PreprocessingStepConfig,
    SplitConfig,
    TaskType,
)
from broadway.features import recipe as recipe_module
from broadway.training import hpo
from broadway.training import hpo as hpo_module
from broadway.training.hpo import (
    bandit_allocate,
    log_best_artifacts,
    make_objective,
    run_hpo,
    run_model_study,
)
from broadway.training.mlflow_utils import setup_mlflow
from broadway.training.trainer import build_model_pipeline


def _pipeline_config(
    preprocessing: list[PreprocessingStepConfig] | None = None,
) -> PipelineConfig:
    """Minimal PipelineConfig for synthetic HPO tests (no preprocessing by default)."""
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir="data",
        raw_subdir="raw",
        processed_subdir="processed",
        download_chunk_size=8192,
        mlflow_tracking_uri="mlruns",
        database_user="user",
        database_password="pass",
        database_name="db",
        database_host="localhost",
        database_port=5432,
        sample_size_ci=1000,
        sample_size_stats=10000,
        api_replicas_min=1,
        api_replicas_max=3,
        api_hpa_cpu_threshold=80,
        monitoring_schedule="0 * * * *",
    )
    dataset = DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "cat": ColumnSchema(dtype="object", null_count=0, role=ColumnRole.FEATURE),
            "num": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        features=FeatureConfig(include=["cat", "num"], exclude=[], derived=[], encodings=[]),
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        preprocessing=preprocessing or [],
    )
    return PipelineConfig(dataset=dataset, environment=environment, experiment=experiment)


def _parabola(params: dict[str, float | int], trial=None) -> float:
    del trial
    return float((params["x"] - 3.0) ** 2)


def _hpo_config(
    n_models: int = 2,
    total_trials: int = 50,
    initial: int = 10,
    top_k: int = 1,
) -> HPOConfig:
    return HPOConfig(
        engine="optuna",
        direction="minimize",
        total_trials=total_trials,
        initial_trials_per_model=initial,
        top_k=top_k,
        target_metric="rmse",
        models=[
            ModelHPOSpec(name=f"m{index}", search_space={"x": [-10.0, 10.0]})
            for index in range(n_models)
        ],
    )


@pytest.fixture
def tiny_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0])
    X_val = pd.DataFrame({"a": [5.0, 6.0]})
    y_val = pd.Series([10.0, 12.0])
    return X_train, y_train, X_val, y_val


# --- bandit_allocate ---------------------------------------------------------


def test_bandit_allocate_top_k_selection() -> None:
    leaderboard = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert bandit_allocate(leaderboard, 9, 2) == {"a": 5, "b": 4}


def test_bandit_allocate_even_split() -> None:
    assert bandit_allocate({"lgbm": 1.0, "xgb": 2.0, "ols": 3.0}, 80, 2) == {
        "lgbm": 40,
        "xgb": 40,
    }


def test_bandit_allocate_remainder_to_best() -> None:
    assert bandit_allocate({"a": 1.0, "b": 2.0}, 5, 2) == {"a": 3, "b": 2}


def test_bandit_allocate_remaining_less_than_k() -> None:
    assert bandit_allocate({"a": 1.0, "b": 2.0, "c": 3.0}, 2, 5) == {"a": 1, "b": 1, "c": 0}


def test_bandit_allocate_top_k_beyond_models() -> None:
    assert bandit_allocate({"a": 1.0, "b": 2.0}, 4, 10) == {"a": 2, "b": 2}


def test_bandit_allocate_empty_leaderboard() -> None:
    assert bandit_allocate({}, 10, 2) == {}


def test_bandit_allocate_no_remaining() -> None:
    assert bandit_allocate({"a": 1.0, "b": 2.0}, 0, 2) == {}


def test_bandit_allocate_zero_top_k() -> None:
    assert bandit_allocate({"a": 1.0}, 10, 0) == {}


# --- make_objective ----------------------------------------------------------


def test_make_objective_returns_target_metric(tiny_data) -> None:
    X_train, y_train, X_val, y_val = tiny_data
    cfg = _pipeline_config()
    objective = make_objective(cfg, "linear", "rmse", X_train, y_train, X_val, y_val)
    assert objective({}) == pytest.approx(0.0, abs=0.1)
    objective_r2 = make_objective(cfg, "linear", "r2", X_train, y_train, X_val, y_val)
    assert objective_r2({}) == pytest.approx(1.0, abs=0.05)


# --- run_model_study ---------------------------------------------------------


def test_run_model_study_sqlite_reopens_and_accumulates(tmp_path) -> None:
    storage_url = f"sqlite:///{tmp_path}/study.db"
    spec = ModelHPOSpec(name="m0", search_space={"x": [-10.0, 10.0]})
    first = run_model_study(spec, _parabola, n_trials=10, random_state=42, storage_url=storage_url)
    assert len(first.trials) == 10
    reopened = run_model_study(spec, _parabola, n_trials=5, random_state=7, storage_url=storage_url)
    assert len(reopened.trials) == 15
    assert reopened.best_params["x"] == pytest.approx(3.0, abs=1.0)
    assert reopened.best_value == pytest.approx(0.0, abs=1.0)


# --- run_hpo -----------------------------------------------------------------


class _ImmediateExecutor:
    """Runs submitted work eagerly — sequential stand-in for ThreadPoolExecutor."""

    def __init__(self, max_workers: int | None = None) -> None:
        del max_workers

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn, *args, **kwargs):
        class _Future:
            def __init__(self, value: object) -> None:
                self._value = value

            def result(self) -> object:
                return self._value

        return _Future(fn(*args, **kwargs))


def _stub_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hpo_module, "make_objective", lambda *args, **kwargs: _parabola)


def test_run_hpo_end_to_end(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    result = run_hpo(_pipeline_config(), _hpo_config(), X_train, y_train, X_val, y_val, random_state=42)
    assert result["best_params"]["x"] == pytest.approx(3.0, abs=1.0)
    assert result["best_value"] == pytest.approx(0.0, abs=1.0)
    assert result["best_model"] in {"m0", "m1"}
    # total budget: initial (2 * 10) + remaining 30 to the single top-k model
    assert sum(model["n_trials"] for model in result["models"].values()) == 50
    assert max(model["n_trials"] for model in result["models"].values()) == 40


def test_run_hpo_deterministic(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    first = run_hpo(_pipeline_config(), _hpo_config(), X_train, y_train, X_val, y_val, random_state=7)
    second = run_hpo(_pipeline_config(), _hpo_config(), X_train, y_train, X_val, y_val, random_state=7)
    assert first == second


def test_run_hpo_parallel_equals_sequential(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    parallel = run_hpo(_pipeline_config(), _hpo_config(n_models=3, top_k=2), X_train, y_train, X_val, y_val, random_state=3)
    monkeypatch.setattr(hpo_module, "ThreadPoolExecutor", _ImmediateExecutor)
    sequential = run_hpo(_pipeline_config(), _hpo_config(n_models=3, top_k=2), X_train, y_train, X_val, y_val, random_state=3)
    assert parallel == sequential


def test_run_hpo_no_bandit_when_budget_exhausted(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    result = run_hpo(_pipeline_config(), _hpo_config(total_trials=20, initial=10), X_train, y_train, X_val, y_val, random_state=1)
    assert sum(model["n_trials"] for model in result["models"].values()) == 20


def test_run_hpo_requires_models(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    hpo = _hpo_config(n_models=0)
    with pytest.raises(ValueError, match="at least one model"):
        run_hpo(_pipeline_config(), hpo, X_train, y_train, X_val, y_val, random_state=1)


def test_run_hpo_no_valid_trial_raises(tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    def _nan(params: dict[str, float | int], trial=None) -> float:
        del params, trial
        return float("nan")

    monkeypatch.setattr(hpo_module, "make_objective", lambda *args, **kwargs: _nan)
    X_train, y_train, X_val, y_val = tiny_data
    with pytest.raises(ValueError, match="no valid trial"):
        run_hpo(_pipeline_config(), _hpo_config(), X_train, y_train, X_val, y_val, random_state=1)


# --- mlflow tracking (hermetic tmp file store, no server) -------------------


def _mlflow_file_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point mlflow at a per-test tmp file store."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")


def test_mlflow_tracking_logs_trials(tmp_path, tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _mlflow_file_store(tmp_path, monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    objective = make_objective(_pipeline_config(), "linear", "rmse", X_train, y_train, X_val, y_val)
    spec = ModelHPOSpec(name="m0", search_space={"n_jobs": [1, 4]})
    n_trials = 5
    run_model_study(spec, objective, n_trials=n_trials, random_state=42,
                    mlflow_tracking=True, mlflow_tags={"model": "m0"})
    runs = mlflow.search_runs(experiment_names=["test_experiment"])
    assert len(runs) == n_trials
    assert (runs["status"] == "FINISHED").all()
    for _, run in runs.iterrows():
        assert run["tags.study"] == "hpo-m0"
        assert run["tags.model"] == "m0"
        assert run["tags.trial"] in {str(i) for i in range(n_trials)}
        assert run["tags.mlflow.runName"].startswith("hpo-m0 trial ")
        assert run["params.n_jobs"] in {"1", "2", "3", "4"}
        assert run["metrics.rmse"] == pytest.approx(0.0, abs=0.1)
        assert run["metrics.mae"] == pytest.approx(0.0, abs=0.1)
        assert run["metrics.r2"] == pytest.approx(1.0, abs=0.05)
        assert run["metrics.target_metric"] == pytest.approx(run["metrics.rmse"])


def test_run_hpo_mlflow_tags(tmp_path, tiny_data, monkeypatch: pytest.MonkeyPatch) -> None:
    _mlflow_file_store(tmp_path, monkeypatch)
    _stub_objective(monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    run_hpo(_pipeline_config(), _hpo_config(), X_train, y_train, X_val, y_val, random_state=42,
            mlflow_tracking=True, mlflow_tags={"exp": "x"})
    runs = mlflow.search_runs(experiment_names=["test_experiment"])
    assert len(runs) == 50
    assert set(runs["tags.model"]) == {"m0", "m1"}
    assert set(runs["tags.study"]) == {"hpo-m0", "hpo-m1"}
    assert (runs["tags.exp"] == "x").all()


def _run_artifacts(run_id: str) -> tuple[list[str], bool]:
    """Run artifact paths, plus whether it carries a logged model output."""
    client = mlflow.tracking.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    outputs = client.get_run(run_id).outputs
    has_model = outputs is not None and len(outputs.model_outputs) == 1
    return artifacts, has_model


def test_log_best_artifacts_linear_model_and_csv(tmp_path, tiny_data,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _mlflow_file_store(tmp_path, monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    with mlflow.start_run():
        log_best_artifacts(_pipeline_config(), "linear", {}, X_train, y_train, X_val, y_val)
    runs = mlflow.search_runs(experiment_names=["test_experiment"])
    assert len(runs) == 1
    artifacts, has_model = _run_artifacts(runs.iloc[0]["run_id"])
    assert has_model
    assert "predictions.csv" in artifacts
    assert "feature_importance.png" not in artifacts


def test_log_best_artifacts_tree_importance_plot(tmp_path, tiny_data,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _mlflow_file_store(tmp_path, monkeypatch)
    X_train, y_train, X_val, y_val = tiny_data
    with mlflow.start_run():
        log_best_artifacts(_pipeline_config(), "lgbm", {"n_estimators": 5, "max_depth": 2},
                           X_train, y_train, X_val, y_val)
    runs = mlflow.search_runs(experiment_names=["test_experiment"])
    assert len(runs) == 1
    artifacts, has_model = _run_artifacts(runs.iloc[0]["run_id"])
    assert has_model
    assert "predictions.csv" in artifacts
    assert "feature_importance.png" in artifacts


def test_make_objective_returns_float() -> None:
    """The objective must return a finite, non-negative float on synthetic data.

    The objective receives a params dict (the run_study contract); passing {}
    exercises the default-params path, and a small hyperparam set exercises
    the tuned path — no exceptions on either.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = pd.Series(2.0 * X["a"] + 0.5 + rng.normal(scale=0.1, size=40))
    for model_type in ("linear", "lgbm", "xgb"):
        objective = hpo.make_objective(_pipeline_config(), model_type, "mae", X, y, X, y)
        value = objective({})
        assert isinstance(value, float) and np.isfinite(value) and value >= 0.0
        tuned = objective({"n_estimators": 10, "max_depth": 2} if model_type != "linear" else {})
        assert isinstance(tuned, float) and np.isfinite(tuned)


def test_hpo_trial_refits_preprocessing_each_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leakage guard: make_objective fits the composed Pipeline per trial, so
    preprocessing refits on train data every trial — no fit-once reuse of
    transform state. HPO uses a single train/val split (no internal CV), so
    per-trial refit IS the complete guard."""
    shared = {"fits": 0}

    class _CountingTransformer(BaseEstimator):
        """Target-encoding stand-in: counts refits and maps categories to numbers."""

        def __init__(self, state: dict) -> None:
            self._state = state

        def fit(self, X: pd.DataFrame, y=None):
            self._state["fits"] += 1
            self.mapping_ = {
                value: float(i) for i, value in enumerate(sorted(set(X["cat"])))
            }
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return X.assign(cat=X["cat"].map(self.mapping_))

    def counting_build_step(step, target):
        if step.type == "target_encoding":
            return _CountingTransformer(shared)
        raise AssertionError(f"unexpected step type in leakage test: {step.type}")

    monkeypatch.setattr(recipe_module, "_build_step", counting_build_step)
    cfg = _pipeline_config(
        [
            PreprocessingStepConfig(
                type="target_encoding", columns=["cat"], params={"smoothing": 20}
            )
        ]
    )
    X_train = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0])
    X_val = pd.DataFrame({"cat": ["a", "b"], "num": [5.0, 6.0]})
    y_val = pd.Series([10.0, 12.0])
    objective = make_objective(cfg, "linear", "rmse", X_train, y_train, X_val, y_val)
    assert objective({}) == pytest.approx(0.0, abs=0.1)
    assert objective({}) == pytest.approx(0.0, abs=0.1)
    assert shared["fits"] == 2


def test_hyperopt_smoke_through_pipeline_objective(tiny_data) -> None:
    """Hyperopt smoke: a study over a Pipeline objective (one_hot recipe +
    linear model) completes and returns a finite best value."""
    cfg = _pipeline_config(
        [PreprocessingStepConfig(type="one_hot", columns=["cat"], params={})]
    )
    X_train = pd.DataFrame(
        {"cat": ["a", "b", "a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    X_val = pd.DataFrame({"cat": ["a", "b"], "num": [7.0, 8.0]})
    y_val = pd.Series([14.0, 16.0])
    objective = make_objective(cfg, "linear", "mae", X_train, y_train, X_val, y_val)
    spec = ModelHPOSpec(name="m0", search_space={"n_jobs": [1, 2]})
    study = run_model_study(spec, objective, n_trials=3, random_state=42)
    assert study.best_value is not None and np.isfinite(study.best_value)


def test_pre_params_reach_pipeline_params() -> None:
    """pre__<step>__<param> keys reach the Pipeline via get_params after
    build_model_pipeline — the HPO search-space addressing contract."""
    cfg = _pipeline_config(
        [
            PreprocessingStepConfig(
                type="target_encoding", columns=["cat"], params={"smoothing": 20}
            )
        ]
    )
    pipeline = build_model_pipeline(
        cfg, "lgbm", {"n_estimators": 10, "pre__target_encoding_0__smoothing": 35}
    )
    params = pipeline.get_params()
    assert params["pre__target_encoding_0__smoothing"] == 35
    assert params["model__n_estimators"] == 10
