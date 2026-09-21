"""Extended tests for the MLflow thin wrappers: metadata and dataset lineage."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from broadway.config.schema import PipelineConfig
from broadway.training.mlflow_utils import (
    log_dataset,
    log_datasets,
    log_metadata,
    log_model,
    setup_mlflow,
)

_LOGGER_NAME = "broadway.training.mlflow_utils"


def test_log_metadata_records_metrics(tmp_path: Path) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    with mlflow.start_run() as run:
        log_metadata({"train_time_seconds": 1.5, "model_size_bytes": 4096.0})
    run_data = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data
    assert run_data.metrics["train_time_seconds"] == pytest.approx(1.5)
    assert run_data.metrics["model_size_bytes"] == pytest.approx(4096.0)


def test_log_dataset_records_param_and_lineage(tmp_path: Path) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    source = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(source, index=False)
    with mlflow.start_run() as run:
        log_dataset("test-train", str(source), context="train")
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-train"
    inputs = run_obj.inputs.dataset_inputs
    assert len(inputs) == 1
    assert str(source) in inputs[0].dataset.source
    tags = {tag.key: tag.value for tag in inputs[0].tags}
    assert tags["mlflow.data.context"] == "train"


def test_log_dataset_missing_source_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    missing = tmp_path / "missing.parquet"
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_dataset("test-train", str(missing), context="train")
    assert "missing.parquet" in caplog.text
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-train"
    assert run_obj.inputs.dataset_inputs == []


def test_log_datasets_logs_train_and_eval_without_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D9: in-memory frames produce train+eval lineage with no parquet re-read."""
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    train_source = tmp_path / "train.parquet"
    val_source = tmp_path / "val.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(train_source, index=False)
    pd.DataFrame({"a": [7, 8], "b": [9.0, 10.0]}).to_parquet(val_source, index=False)
    train_df = pd.read_parquet(train_source)
    val_df = pd.read_parquet(val_source)

    def _no_reread(*args: object, **kwargs: object) -> object:
        raise AssertionError("log_datasets must not re-read parquet from disk")

    monkeypatch.setattr(pd, "read_parquet", _no_reread)
    with mlflow.start_run() as run:
        log_datasets("test-id", train_df, str(train_source), val_df, str(val_source))
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-id"
    inputs = run_obj.inputs.dataset_inputs
    assert len(inputs) == 2
    contexts = {tag.value for inp in inputs for tag in inp.tags if tag.key == "mlflow.data.context"}
    assert contexts == {"train", "eval"}
    sources = [inp.dataset.source for inp in inputs]
    assert any(str(train_source) in s for s in sources)
    assert any(str(val_source) in s for s in sources)


def test_log_datasets_missing_val_source_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D9: a missing val source warns per-frame; train lineage still lands."""
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    train_source = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(train_source, index=False)
    train_df = pd.read_parquet(train_source)
    val_df = pd.DataFrame({"a": [4, 5]})
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_datasets("test-id", train_df, str(train_source), val_df, str(tmp_path / "nope.parquet"))
    assert "skipping lineage" in caplog.text
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-id"
    inputs = run_obj.inputs.dataset_inputs
    assert len(inputs) == 1
    tags = {tag.key: tag.value for tag in inputs[0].tags}
    assert tags["mlflow.data.context"] == "train"


def test_log_datasets_none_val_skips_quietly(tmp_path: Path) -> None:
    """D9: val_df=None (no val split) logs train lineage only, no warning."""
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    train_source = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(train_source, index=False)
    train_df = pd.read_parquet(train_source)
    with mlflow.start_run() as run:
        log_datasets("test-id", train_df, str(train_source))
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert len(run_obj.inputs.dataset_inputs) == 1


def test_module_run_wires_in_memory_frames_without_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: module.run logs train+eval lineage on exactly 2 parquet reads."""
    from broadway.lineage import records
    from broadway.training import module

    cfg = _training_config(tmp_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    out_dir = Path(cfg.environment.data_dir) / cfg.environment.processed_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "rooms": np.arange(1, 41),
            "area": np.arange(40, 80),
            "price": np.arange(1, 41) * 100.0,
        }
    )
    df.to_parquet(out_dir / cfg.etl.train_features_file, index=False)
    df.tail(10).to_parquet(out_dir / cfg.etl.val_features_file, index=False)

    real_read_parquet = pd.read_parquet
    reads = {"count": 0}

    def _counting_read(*args: object, **kwargs: object) -> pd.DataFrame:
        reads["count"] += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _counting_read)
    module.run(cfg)

    # The only parquet reads are the two feature loads — lineage reuses frames.
    assert reads["count"] == 2
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[client.get_experiment_by_name("synthetic").experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["dataset_id"] == "synthetic"
    contexts = {
        tag.value
        for inp in runs[0].inputs.dataset_inputs
        for tag in inp.tags
        if tag.key == "mlflow.data.context"
    }
    assert contexts == {"train", "eval"}


def _training_config(tmp_path: Path) -> PipelineConfig:
    from broadway.analysis.contracts import AnalysisContract, AnalysisMode
    from broadway.config.schema import (
        ColumnRole,
        ColumnSchema,
        DatasetContract,
        DataSourceRef,
        EnvironmentConfig,
        EtlStep,
        ExperimentConfig,
        FeatureConfig,
        ModelConfig,
        SplitConfig,
        TaskType,
        TrainStep,
    )

    return PipelineConfig(
        dataset=DatasetContract(
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
        ),
        environment=EnvironmentConfig(
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
        ),
        experiment=ExperimentConfig(
            data_source=DataSourceRef(loader="canonical", schema_contract="raw"),
            features=FeatureConfig(include=["rooms", "area"], exclude=[], derived=[], encodings=[]),
            model=ModelConfig(type="linear", params={}),
            split=SplitConfig(type="random", validation_size=0.2),
            random_state=42,
            target_metric="rmse",
            hpo=None,
        ),
        etl=EtlStep(
            ci_sample_size=0,
            max_drop_fraction=0.5,
            random_state=42,
            train_file="train.parquet",
            val_file="val.parquet",
            training_data_file="training_data.parquet",
            train_features_file="train_features.parquet",
            val_features_file="val_features.parquet",
            missing_encodings=["", "NA", "null"],
        ),
        train=TrainStep(
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
            output_dir=str(tmp_path / "artifacts" / "training"),
            output_file="training_result.json",
        ),
        analysis=AnalysisContract(
            name="test",
            mode=AnalysisMode.PREDICTION,
            goal="predict price",
            row_definition="one row",
            decision_moment="now",
            available_info=["rooms"],
            leakage_notes=[],
            success_criterion="beat baseline",
        ),
    )


def test_logged_pipeline_reloads_and_predicts_raw_frame(tmp_path: Path) -> None:
    """The logged artifact is the Pipeline and carries the signature (Slice 3).

    Logs a fitted Pipeline with an explicit signature, reloads it via the
    sklearn flavor, and asserts it predicts on a raw feature frame — proving
    preprocessing (or passthrough) ships inside the artifact, not the bare
    model alone.
    """
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    y = pd.Series([3.0, 6.0, 9.0])
    pipeline = Pipeline([("model", LinearRegression())]).fit(X, y)
    with mlflow.start_run():
        uri = log_model(pipeline, "model", signature=infer_signature(X, y))
    loaded = mlflow.sklearn.load_model(uri)
    preds = loaded.predict(X)
    assert preds.shape == (3,)
    assert np.issubdtype(preds.dtype, np.floating)
    assert mlflow.models.get_model_info(uri).signature is not None


def test_local_source_missing_registry_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """_local_source maps an empty registry to a fail-loud RuntimeError."""
    from broadway.training import mlflow_utils as mlflow_utils_module

    monkeypatch.setattr(mlflow_utils_module, "get_registered_sources", list)
    with pytest.raises(RuntimeError, match="no longer registers"):
        mlflow_utils_module._local_source("/tmp/nowhere.parquet")


def test_log_datasets_missing_train_source_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D9: a missing train source warns per-frame; eval lineage still lands."""
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    val_source = tmp_path / "val.parquet"
    pd.DataFrame({"a": [7, 8]}).to_parquet(val_source, index=False)
    val_df = pd.read_parquet(val_source)
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_datasets("test-id", train_df, str(tmp_path / "nope.parquet"), val_df, str(val_source))
    assert "skipping lineage" in caplog.text
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert run_obj.data.params["dataset_id"] == "test-id"
    inputs = run_obj.inputs.dataset_inputs
    assert len(inputs) == 1
    tags = {tag.key: tag.value for tag in inputs[0].tags}
    assert tags["mlflow.data.context"] == "eval"


def test_log_datasets_none_val_skips_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """D9: val_df=None pins the quiet skip — train lands, no warning."""
    setup_mlflow(str(tmp_path / "mlruns"), "test_experiment")
    train_source = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(train_source, index=False)
    train_df = pd.read_parquet(train_source)
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        log_datasets("test-id", train_df, str(train_source), None, None)
    assert "skipping lineage" not in caplog.text
    run_obj = mlflow.tracking.MlflowClient().get_run(run.info.run_id)
    assert len(run_obj.inputs.dataset_inputs) == 1


def _etl_frame(feats: list[str], target: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            feats[1]: [100, 100, 150, 200],
            feats[2]: ["a", "a", "b", "c"],
            target: [10, 10, 30, 40],
            feats[0]: [2, 2, 3, 4],
        }
    )


def test_etl_run_writes_data_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Data-tracking lane: etl run() reaches the manifest hook and writes it."""
    from contract_fixture import frame_slots

    import broadway.etl.module as etl_module
    from broadway.config.loader import load_config
    from broadway.lineage import records

    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    cfg = cfg.model_copy(
        update={"environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)})}
    )
    assert cfg.dataset is not None
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setenv("BROADWAY_TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("BROADWAY_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    feats, target = frame_slots(cfg.dataset)
    df = _etl_frame(list(feats), target)
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (df.copy(), [], []))

    etl_module.run(cfg)

    assert (tmp_path / "tracking" / "data_manifest.json").exists()


def test_etl_run_without_experiment_fails_loud_at_manifest_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Data-tracking lane: the manifest hook fails loud when experiment is None."""
    from contract_fixture import frame_slots

    import broadway.etl.module as etl_module
    from broadway.config.loader import load_config
    from broadway.lineage import records

    cfg = load_config("etl", dataset="test", experiment="baseline")
    assert cfg.dataset is not None
    cfg = cfg.model_copy(
        update={
            "environment": cfg.environment.model_copy(update={"data_dir": str(tmp_path)}),
            "experiment": None,
        }
    )
    assert cfg.dataset is not None
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    # Bypass the entry guard so execution reaches the manifest hook itself.
    monkeypatch.setattr(etl_module, "_assert_data_source_supported", lambda cfg: None)

    feats, target = frame_slots(cfg.dataset)
    df = _etl_frame(list(feats), target)
    monkeypatch.setattr(etl_module, "load_with_audit", lambda dataset: (df.copy(), [], []))

    with pytest.raises(ValueError, match="etl manifest requires an experiment binding"):
        etl_module.run(cfg)
