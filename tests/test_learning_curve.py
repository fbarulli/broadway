from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    ExperimentConfig,
    LearningCurveConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    TaskType,
)
from broadway.evaluate.metrics import binary_metrics
from broadway.features.plan import resolve_plan
from broadway.training.learning_curve import Direction, LearningCurve


def test_learning_curve_orders_points_and_reports_minimum_data(tmp_path: Path) -> None:
    """A noiseless linear target fits at every fraction, so the smallest
    fraction meets the full-data reference; the curve is ordered, rows grow,
    the frames are untouched, and the JSON/CSV/PNG artifacts are written."""
    cfg = _config(tmp_path)
    rng = np.random.default_rng(7)
    rooms = rng.integers(1, 9, size=40).astype("int64")
    area = rng.integers(20, 120, size=40).astype("int64")
    price = 3.0 * rooms + 2.0 * area
    train_df = pd.DataFrame({"rooms": rooms, "area": area, "price": price})
    val_df = train_df.iloc[:10].copy()
    before = train_df.copy()

    config = _curve_config(tmp_path, fractions=[0.25, 0.5, 1.0], repeats=2)
    result = LearningCurve(cfg, config, lambda: Pipeline([("model", LinearRegression())])).run(
        train_df, val_df
    )

    assert [point.fraction for point in result.points] == [0.25, 0.5, 1.0]
    assert [point.rows for point in result.points] == [10, 20, 40]
    assert result.reference.fraction == 1.0
    assert all(point.within_tolerance for point in result.points)
    assert result.min_fraction == 0.25
    assert result.min_rows == 10
    assert train_df.equals(before)

    assert config.output_path.exists()
    assert config.csv_path.exists()
    assert config.png_path.exists() and config.png_path.stat().st_size > 0
    written = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert written["min_fraction"] == 0.25
    assert [point["fraction"] for point in written["points"]] == [0.25, 0.5, 1.0]


def test_learning_curve_classification_uses_auc_path(tmp_path: Path) -> None:
    """A classification target scores through binary_metrics (roc_auc) with the
    configured threshold, and the stratified draw keeps both classes."""
    cfg = _config(tmp_path, task=TaskType.CLASSIFICATION)
    rng = np.random.default_rng(11)
    rooms = rng.integers(1, 10, size=100).astype("int64")
    area = rng.integers(20, 120, size=100).astype("int64")
    label = (rooms >= 5).astype("int64")
    train_df = pd.DataFrame({"rooms": rooms, "area": area, "label": label})
    val_df = train_df.iloc[:20].copy()

    config = _curve_config(
        tmp_path,
        fractions=[0.25, 0.5, 1.0],
        repeats=1,
        direction="maximize",
        metric_threshold=0.5,
    )
    result = LearningCurve(
        cfg, config, lambda: Pipeline([("model", LogisticRegression(max_iter=1000))])
    ).run(train_df, val_df)

    assert result.target_metric == "roc_auc"
    assert result.split == "stratified"
    assert all(point.mean is not None and 0.0 <= point.mean <= 1.0 for point in result.points)
    assert result.min_fraction == 0.25


def test_auc_uses_probabilities_and_requires_predict_proba(tmp_path: Path) -> None:
    """AUC is computed from positive-class probabilities, not labels, and a
    model without predict_proba fails loud instead of scoring labels."""
    cfg = _config(tmp_path, task=TaskType.CLASSIFICATION)
    rng = np.random.default_rng(1)
    n = 200
    rooms = rng.normal(size=n)
    area = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(1.2 * rooms + 0.8 * area)))
    label = (rng.random(n) < probability).astype("int64")
    frame = pd.DataFrame({"rooms": rooms, "area": area, "label": label})
    train_df, val_df = frame.iloc[:150], frame.iloc[150:]
    y_true = val_df["label"].to_numpy()

    reference = Pipeline([("model", LogisticRegression(max_iter=1000))])
    reference.fit(train_df[["rooms", "area"]], train_df["label"])
    proba_auc = binary_metrics(
        y_true, reference.predict_proba(val_df[["rooms", "area"]])[:, 1], 0.5
    )["roc_auc"]
    label_auc = binary_metrics(y_true, reference.predict(val_df[["rooms", "area"]]), 0.5)["roc_auc"]
    assert proba_auc != label_auc

    result = LearningCurve(
        cfg,
        _curve_config(
            tmp_path,
            fractions=[1.0],
            direction="maximize",
            metric_threshold=0.5,
        ),
        lambda: Pipeline([("model", LogisticRegression(max_iter=1000))]),
    ).run(train_df, val_df)
    assert result.reference.mean == proba_auc

    with pytest.raises(ValueError, match="roc_auc.*linear.*predict_proba"):
        LearningCurve(
            cfg,
            _curve_config(tmp_path, fractions=[1.0], metric_threshold=0.5),
            lambda: Pipeline([("model", LinearRegression())]),
        ).run(train_df, val_df)


def test_absolute_tolerance_floor_rescues_a_zero_reference(tmp_path: Path) -> None:
    """With a (near-)zero full-data metric the relative band would demand exact
    equality; the absolute floor is what admits a smaller fraction."""
    cfg = _config(tmp_path)
    x = np.arange(1, 61)
    train_df = pd.DataFrame(
        {"rooms": x.astype("int64"), "area": (x % 7).astype("int64"), "price": x**2.0}
    )
    val_df = train_df.iloc[:20].copy()

    def factory() -> Pipeline:
        return Pipeline([("model", LinearRegression())])

    strict = LearningCurve(
        cfg,
        _curve_config(tmp_path / "strict", fractions=[0.1, 1.0], direction="minimize"),
        factory,
    ).run(train_df, val_df)
    floored = LearningCurve(
        cfg,
        _curve_config(
            tmp_path / "floored",
            fractions=[0.1, 1.0],
            direction="minimize",
            absolute_tolerance=1e6,
        ),
        factory,
    ).run(train_df, val_df)

    assert strict.min_fraction == 1.0
    assert floored.min_fraction == 0.1


def _curve_config(
    tmp_path: Path,
    *,
    fractions: list[float],
    repeats: int = 1,
    direction: Direction = "minimize",
    absolute_tolerance: float = 0.0,
    metric_threshold: float | None = None,
) -> LearningCurveConfig:
    return LearningCurveConfig(
        fractions=fractions,
        repeats=repeats,
        direction=direction,
        tolerance=0.01,
        absolute_tolerance=absolute_tolerance,
        metric_threshold=metric_threshold,
        output_dir=str(tmp_path / "artifacts"),
        output_file="learning_curve.json",
    )


def _config(tmp_path: Path, *, task: TaskType = TaskType.REGRESSION) -> PipelineConfig:
    classification = task is TaskType.CLASSIFICATION
    target = "label" if classification else "price"
    target_dtype = "int64" if classification else "float64"
    environment = EnvironmentConfig(
        log_level="INFO",
        data_dir=str(tmp_path / "data"),
        raw_subdir="raw",
        processed_subdir="processed",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
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
    )
    dataset = DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target=target,
        task=task,
        datetime_column=None,
        columns={
            "rooms": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            target: ColumnSchema(dtype=target_dtype, null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        feature_plan="broadway.features.plans:contract_numeric_features",
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="stratified" if classification else "random", validation_size=0.2),
        random_state=42,
        target_metric="roc_auc" if classification else "rmse",
    )
    experiment.bind_plan(resolve_plan(experiment.feature_plan, dataset))
    return PipelineConfig(
        environment=environment,
        dataset=dataset,
        experiment=experiment,
        analysis=AnalysisContract(
            name="test",
            mode=AnalysisMode.PREDICTION,
            goal="predict target",
            row_definition="one row",
            decision_moment="now",
            available_info=["rooms"],
            leakage_notes=[],
            success_criterion="beat baseline",
        ),
    )
