"""Public behavior of the selection/hpo/training-decision steps (synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.analysis.contracts import AnalysisContract, AnalysisMode
from broadway.baseline import module as baseline_module
from broadway.config.schema import (
    BaselineStep,
    BootstrapConfig,
    CandidateSpec,
    ColumnRole,
    ColumnSchema,
    CVGeometry,
    DatasetContract,
    DataSourceRef,
    EnvironmentConfig,
    EtlStep,
    ExperimentConfig,
    HPOConfig,
    HpoStep,
    ModelConfig,
    ModelHPOSpec,
    PipelineConfig,
    SelectionStep,
    SplitConfig,
    TaskType,
    TrainStep,
)
from broadway.evaluate.repeated_cv import RepeatedCV
from broadway.features.plan import resolve_plan
from broadway.lineage import records
from broadway.reports.decision import Decision
from broadway.selection import module as selection_module
from broadway.selection import tuning
from broadway.selection.candidates import CandidateSet
from broadway.training import module as training_module


def _environment(tmp_path: Path) -> EnvironmentConfig:
    return EnvironmentConfig(
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


def _dataset() -> DatasetContract:
    return DatasetContract(
        name="synthetic",
        path="synthetic.parquet",
        target="price",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "rooms": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "area": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.FEATURE),
            "price": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={},
    )


def _experiment(hpo: HPOConfig | None = None) -> ExperimentConfig:
    experiment = ExperimentConfig(
        data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
        feature_plan="broadway.features.plans:contract_numeric_features",
        model=ModelConfig(type="linear", params={}),
        split=SplitConfig(type="random", validation_size=0.2),
        random_state=42,
        target_metric="rmse",
        hpo=hpo,
    )
    return experiment.bind_plan(resolve_plan(experiment.feature_plan, _dataset()))


def _etl() -> EtlStep:
    return EtlStep(
        ci_sample_size=0,
        max_drop_fraction=0.5,
        random_state=42,
        train_file="train.parquet",
        val_file="val.parquet",
        training_data_file="training_data.parquet",
        train_features_file="train_features.parquet",
        val_features_file="val_features.parquet",
        missing_encodings=[""],
    )


def _write_features(tmp_path: Path, rows: int = 80, val_rows: int = 0) -> None:
    directory = tmp_path / "data" / "processed"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rooms = rng.integers(1, 6, rows + val_rows)
    area = rng.integers(40, 120, rows + val_rows)
    price = 3.0 * rooms + 2.0 * area + rng.normal(0, 1.0, rows + val_rows)
    frame = pd.DataFrame({"rooms": rooms, "area": area, "price": price})
    frame.iloc[:rows].to_parquet(directory / "train_features.parquet", index=False)
    if val_rows:
        frame.iloc[rows:].to_parquet(directory / "val_features.parquet", index=False)


def _selection_step(tmp_path: Path) -> SelectionStep:
    return SelectionStep(
        metric="rmse",
        secondary_metric="mae",
        metric_direction="min",
        secondary_direction="min",
        cv=CVGeometry(folds=3, repeats=2, random_state=5),
        bootstrap=BootstrapConfig(iterations=200, seed=1, level=0.95),
        sensitivity_random_state=6,
        selection_rule="point",
        baseline_candidate="naive_mean",
        linear_reference="linear",
        candidates=[
            CandidateSpec(name="naive_mean", model="naive_mean", params={}),
            CandidateSpec(name="linear", model="linear", params={}),
        ],
        min_effect=0.01,
        min_sample_size=10,
        action="carry the champion into HPO",
        owner="synthetic",
        output_dir=str(tmp_path / "results" / "selection"),
        output_file=str(tmp_path / "results" / "selection" / "selection_champion.json"),
    )


def test_classification_selection_rejects_the_regression_naive_mean(
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(
        dataset=_dataset().model_copy(update={"task": TaskType.CLASSIFICATION}),
        environment=_environment(tmp_path),
        experiment=_experiment(),
    )
    candidates = [
        CandidateSpec(name="naive_mean", model="naive_mean", params={}),
        CandidateSpec(name="linear", model="linear", params={}),
    ]
    with pytest.raises(ValueError, match="naive_prevalence"):
        CandidateSet(cfg, candidates)


def test_classification_selection_resolves_the_prevalence_reference(
    tmp_path: Path,
) -> None:
    cfg = PipelineConfig(
        dataset=_dataset().model_copy(update={"task": TaskType.CLASSIFICATION}),
        environment=_environment(tmp_path),
        experiment=_experiment(),
    )
    candidates = [
        CandidateSpec(name="naive_prevalence", model="naive_prevalence", params={}),
        CandidateSpec(name="linear", model="linear", params={}),
    ]
    factories = CandidateSet(cfg, candidates).factories()
    estimator = factories["naive_prevalence"]()
    estimator.fit(pd.DataFrame({"rooms": [1, 2, 3, 4]}), pd.Series([0.0, 0.0, 1.0, 1.0]))
    assert estimator.predict(pd.DataFrame({"rooms": [1, 2]})).tolist() == [0.5, 0.5]


def test_grouped_cv_never_places_a_group_in_both_train_and_validation() -> None:
    from sklearn.base import BaseEstimator, RegressorMixin

    seen: list[tuple[set[int], set[int]]] = []

    class _Recorder(BaseEstimator, RegressorMixin):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> _Recorder:
            self._train_groups = set(X["g"].tolist())
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            seen.append((self._train_groups, set(X["g"].tolist())))
            return np.zeros(len(X))

    rng = np.random.default_rng(0)
    rows = 60
    groups = np.repeat(np.arange(20), 3)
    frame = pd.DataFrame({"x": rng.normal(size=rows), "g": groups})
    truth = pd.Series(rng.integers(0, 2, rows).astype(float))
    geometry = CVGeometry(folds=4, repeats=2, random_state=5)
    result = RepeatedCV(geometry, "pr_auc", "f1").run(
        "recorder", _Recorder, frame, truth, pd.Series(groups, name="customer")
    )
    assert result.grouping.startswith("grouped by 'customer'")
    assert seen
    for train_groups, validation_groups in seen:
        assert train_groups.isdisjoint(validation_groups)
    assert not np.isnan(result.out_of_fold_prediction()).any()


def test_selection_publishes_the_champion_and_its_paired_improvement(
    tmp_path: Path,
) -> None:
    _write_features(tmp_path)
    step = _selection_step(tmp_path)
    cfg = PipelineConfig(
        dataset=_dataset(),
        environment=_environment(tmp_path),
        experiment=_experiment(),
        etl=_etl(),
        selection=step,
    )

    selection_module.run(cfg)

    record = Decision.model_validate_json(
        (Path(step.output_dir) / "selection_decision.json").read_text(encoding="utf-8")
    )
    assert record.metric == "improvement_over_naive_rmse"
    assert record.verdict == "act"
    assert record.sample_size == 80
    assert {segment.segment for segment in record.evidence.segments} == {
        "low_target_half",
        "high_target_half",
    }
    champion = json.loads(Path(step.output_file).read_text(encoding="utf-8"))
    assert champion["champion"] == "linear"
    assert champion["model"] == "linear"


def test_hpo_publishes_best_vs_default_on_its_own_folds(tmp_path: Path) -> None:
    _write_features(tmp_path)
    selection = _selection_step(tmp_path)
    Path(selection.output_dir).mkdir(parents=True)
    Path(selection.output_file).write_text(
        json.dumps(
            {
                "champion": "linear",
                "model": "linear",
                "params": {},
                "metric": "rmse",
                "cv_mean": 0.5,
                "interval_low": 0.4,
                "interval_high": 0.6,
                "selection_rule": "point",
                "rule_fired": "point (leaderboard top)",
                "vs_linear_improvement": 0.0,
            }
        ),
        encoding="utf-8",
    )
    hpo = HPOConfig(
        engine="optuna",
        direction="minimize",
        target_metric="rmse",
        total_trials=2,
        initial_trials_per_model=2,
        top_k=1,
        models=[ModelHPOSpec(name="linear", search_space={})],
    )
    step = HpoStep(
        metric="rmse",
        cv=CVGeometry(folds=3, repeats=1, random_state=11),
        report_random_state=99,
        bootstrap=BootstrapConfig(iterations=200, seed=2, level=0.95),
        selection_record=selection.output_file,
        min_effect=0.01,
        min_sample_size=10,
        action="adopt the tuned params",
        owner="synthetic",
        output_dir=str(tmp_path / "results" / "hpo"),
        output_file=str(tmp_path / "results" / "hpo" / "hpo_champion.json"),
    )
    cfg = PipelineConfig(
        dataset=_dataset(),
        environment=_environment(tmp_path),
        experiment=_experiment(hpo),
        etl=_etl(),
        hpo=step,
    )

    tuning.run(cfg)

    record = Decision.model_validate_json(
        (Path(step.output_dir) / "hpo_decision.json").read_text(encoding="utf-8")
    )
    assert record.metric == "improvement_over_default_rmse"
    assert record.sample_size == 80
    tuned = json.loads(Path(step.output_file).read_text(encoding="utf-8"))
    assert tuned["model"] == "linear"
    assert tuned["trials"] == 2


def test_training_fit_decision_publishes_the_train_vs_naive_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_features(tmp_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    train = TrainStep(
        random_state=42,
        n_jobs=1,
        cv_folds=2,
        cv_kind="kfold",
        model_file="model.pkl",
        n_estimators=10,
        learning_rate=0.05,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        quantile_tail=0.9,
        output_dir=str(tmp_path / "results" / "training"),
        output_file="training_result.json",
        fit_decision={
            "metric": "improvement_over_naive_rmse",
            "direction": "max",
            "min_effect": 0.01,
            "min_sample_size": 10,
            "action": "deploy the fit",
            "owner": "synthetic",
        },
        bootstrap=BootstrapConfig(iterations=200, seed=3, level=0.95),
    )
    cfg = PipelineConfig(
        dataset=_dataset(),
        environment=_environment(tmp_path),
        experiment=_experiment(),
        etl=_etl(),
        train=train,
        analysis=AnalysisContract(
            name="synthetic",
            mode=AnalysisMode.PREDICTION,
            goal="predict price",
            row_definition="one row",
            decision_moment="now",
            available_info=["rooms"],
            leakage_notes=[],
            success_criterion="beat naive",
        ),
    )

    training_module.run(cfg)

    record = Decision.model_validate_json(
        (tmp_path / "results" / "training" / "training_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert record.metric == "improvement_over_naive_rmse"
    assert record.verdict == "act"
    assert record.sample_size == 64


def test_baseline_publishes_the_linear_reference_on_the_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_features(tmp_path, rows=100, val_rows=30)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    step = BaselineStep(
        output_dir=str(tmp_path / "results" / "baseline"),
        output_file="baseline.json",
        linear_reference_model="linear",
        bootstrap=BootstrapConfig(iterations=200, seed=7, level=0.95),
        min_effect=0.02,
        min_sample_size=10,
        action="build the model surface",
        owner="synthetic",
    )
    cfg = PipelineConfig(
        dataset=_dataset(),
        environment=_environment(tmp_path),
        experiment=_experiment(),
        etl=_etl(),
        baseline=step,
        analysis=AnalysisContract(
            name="synthetic",
            mode=AnalysisMode.PREDICTION,
            goal="predict price",
            row_definition="one row",
            decision_moment="now",
            available_info=["rooms"],
            leakage_notes=[],
            success_criterion="beat naive",
        ),
    )

    baseline_module.run(cfg)

    record = Decision.model_validate_json(
        (Path(step.output_dir) / "baseline_decision.json").read_text(encoding="utf-8")
    )
    assert record.metric == "improvement_over_naive_mae"
    assert record.verdict == "act"
    assert record.sample_size == 30
    assert record.evidence.provenance["target"] == "ordinal 1-5 treated as interval-scale"
    persisted = json.loads((Path(step.output_dir) / step.output_file).read_text(encoding="utf-8"))
    assert persisted["details"]["holdout_rows"] == 30
    assert persisted["value"] == persisted["details"]["holdout_mean_mae"]
