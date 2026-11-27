"""GAP-1: standalone step configs — the loader attaches sibling sections
declared in _STEP_SECTION_REQUIREMENTS. Synthetic configs only, no project layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from broadway.config import loader
from broadway.config.loader import load_config

_ENVIRONMENT = {
    "log_level": "INFO",
    "data_dir": "data",
    "raw_subdir": "raw",
    "processed_subdir": "processed",
    "mlflow_tracking_uri": "http://localhost:5000",
    "database_user": "postgres",
    "database_password": "postgres",
    "database_name": "broadway",
    "database_host": "localhost",
    "database_port": 5432,
    "sample_size_ci": 1000,
    "sample_size_stats": 10000,
    "api_replicas_min": 1,
    "api_replicas_max": 3,
    "api_hpa_cpu_threshold": 80,
}

_DATASET = {
    "name": "test",
    "path": "demo/demo.csv",
    "target": "target",
    "task": "regression",
    "datetime_column": None,
    "columns": {
        "feature_1": {"dtype": "int64", "null_count": 0, "role": "feature"},
        "target": {"dtype": "int64", "null_count": 0, "role": "target"},
    },
    "lookup_tables": {},
}

_EXPERIMENT = {
    "data_source": {"loader": "canonical", "schema_contract": "raw"},
    "feature_plan": "broadway.features.plans:contract_features",
    "model": {"type": "linear", "params": {}},
    "split": {"type": "random", "validation_size": 0.2},
    "random_state": 42,
    "target_metric": "rmse",
}

_TRAIN = {
    "random_state": 42,
    "n_jobs": -1,
    "cv_folds": 5,
    "cv_kind": "kfold",
    "model_file": "model.pkl",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "quantile_tail": 0.9,
    "output_dir": "artifacts/training",
    "output_file": "training_result.json",
}

_ETL = {
    "ci_sample_size": 50000,
    "max_drop_fraction": 0.1,
    "random_state": 42,
    "train_file": "train.parquet",
    "val_file": "val.parquet",
    "training_data_file": "training_data.parquet",
    "train_features_file": "train_features.parquet",
    "val_features_file": "val_features.parquet",
    "missing_encodings": ["NA"],
}

_EVALUATE = {
    "target_metric": "rmse",
    "promotion_threshold": 0.05,
    "output_dir": "artifacts/evaluation",
    "output_file": "metrics.json",
}

_STEP_OWN = {"train": _TRAIN, "evaluate": _EVALUATE}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _synthetic_configs(
    root: Path, step: str, *, include_etl: bool, include_train: bool
) -> None:
    _write_yaml(root / "environment" / "development.yaml", _ENVIRONMENT)
    _write_yaml(root / "dataset" / "test.yaml", _DATASET)
    _write_yaml(root / "experiment" / "baseline.yaml", _EXPERIMENT)
    _write_yaml(root / "step" / f"{step}.yaml", _STEP_OWN[step])
    if include_etl:
        _write_yaml(root / "step" / "etl.yaml", _ETL)
    if include_train:
        _write_yaml(root / "step" / "train.yaml", _TRAIN)


@pytest.mark.parametrize(
    ("step", "kwargs"),
    [
        ("features", {"dataset": "test", "experiment": "baseline"}),
        ("train", {"dataset": "test", "experiment": "baseline", "analysis": "test"}),
        ("evaluate", {"dataset": "test", "experiment": "baseline", "analysis": "test"}),
        ("baseline", {"dataset": "test", "analysis": "test"}),
    ],
)
def test_standalone_step_loads_declared_sections(step: str, kwargs: dict[str, str]) -> None:
    cfg = load_config(step, **kwargs)
    for section in loader._STEP_SECTION_REQUIREMENTS[step]:
        assert getattr(cfg, section) is not None, f"{step} missing declared section '{section}'"


@pytest.mark.parametrize(
    ("step", "kwargs", "include_etl", "include_train", "missing"),
    [
        ("train", {"dataset": "test", "experiment": "baseline"}, False, True, ["etl"]),
        ("evaluate", {"dataset": "test", "experiment": "baseline"}, True, False, ["train"]),
        ("evaluate", {"dataset": "test", "experiment": "baseline"}, False, False, ["etl", "train"]),
    ],
)
def test_missing_required_step_section_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    step: str,
    kwargs: dict[str, str],
    include_etl: bool,
    include_train: bool,
    missing: list[str],
) -> None:
    _synthetic_configs(tmp_path, step, include_etl=include_etl, include_train=include_train)
    monkeypatch.setattr(loader, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match=step) as exc_info:
        load_config(step, **kwargs)
    message = str(exc_info.value)
    for section in missing:
        assert section in message, f"message must name missing section '{section}': {message}"


def test_config_overlay_precedes_base_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    _write_yaml(base / "environment" / "development.yaml", _ENVIRONMENT)
    _write_yaml(base / "dataset" / "test.yaml", _DATASET)
    _write_yaml(base / "experiment" / "baseline.yaml", _EXPERIMENT)
    _write_yaml(base / "step" / "etl.yaml", _ETL)
    overlay_dataset = {**_DATASET, "path": "overlay/demo.csv"}
    _write_yaml(overlay / "dataset" / "test.yaml", overlay_dataset)

    monkeypatch.setattr(loader, "CONFIGS_DIR", base)
    monkeypatch.setenv(loader.CONFIG_OVERLAY_ENV, str(overlay))

    assert loader.config_path("dataset/test.yaml") == overlay / "dataset" / "test.yaml"
    assert loader.config_path("step/etl.yaml") == base / "step" / "etl.yaml"
    assert load_config("etl", dataset="test", experiment="baseline").dataset.path == "overlay/demo.csv"


def test_step_overlay_layers_a_partial_project_file_over_the_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    _write_yaml(base / "environment" / "development.yaml", _ENVIRONMENT)
    _write_yaml(base / "dataset" / "test.yaml", _DATASET)
    _write_yaml(base / "experiment" / "baseline.yaml", _EXPERIMENT)
    _write_yaml(base / "step" / "etl.yaml", _ETL)
    _write_yaml(overlay / "step" / "etl.yaml", {"ci_sample_size": 1234})

    monkeypatch.setattr(loader, "CONFIGS_DIR", base)
    monkeypatch.setenv(loader.CONFIG_OVERLAY_ENV, str(overlay))

    cfg = load_config("etl", dataset="test", experiment="baseline")

    assert cfg.etl.ci_sample_size == 1234
    assert cfg.etl.max_drop_fraction == _ETL["max_drop_fraction"]


def test_step_overlay_deep_merges_a_nested_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    _write_yaml(base / "environment" / "development.yaml", _ENVIRONMENT)
    _write_yaml(base / "dataset" / "test.yaml", _DATASET)
    _write_yaml(base / "experiment" / "baseline.yaml", _EXPERIMENT)
    _write_yaml(base / "step" / "etl.yaml", _ETL)
    repo_eda = Path(__file__).resolve().parents[1] / "configs" / "step" / "eda.yaml"
    eda_base = yaml.safe_load(repo_eda.read_text(encoding="utf-8"))
    _write_yaml(base / "step" / "eda.yaml", eda_base)
    _write_yaml(
        overlay / "step" / "eda.yaml",
        {"decisions": {"features": {"action": "overlay action"}}},
    )

    monkeypatch.setattr(loader, "CONFIGS_DIR", base)
    monkeypatch.setenv(loader.CONFIG_OVERLAY_ENV, str(overlay))

    cfg = load_config("eda", dataset="test", experiment="baseline")

    assert cfg.eda.decisions.features.action == "overlay action"
    assert cfg.eda.decisions.features.owner == eda_base["decisions"]["features"]["owner"]
    assert (
        cfg.eda.decisions.outliers.action
        == eda_base["decisions"]["outliers"]["action"]
    )


def test_missing_config_overlay_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_overlay = tmp_path / "missing"
    monkeypatch.setenv(loader.CONFIG_OVERLAY_ENV, str(missing_overlay))

    with pytest.raises(FileNotFoundError, match="config overlay directory not found"):
        loader.config_path("dataset/test.yaml")


def _rename_contract(**overrides: object) -> dict:
    contract = {
        "name": "survey",
        "path": "unused.csv",
        "target": "satisfaction",
        "task": "regression",
        "datetime_column": None,
        "columns": {
            "rating_accuracy_score": {"dtype": "int64", "null_count": 0, "role": "feature"},
            "satisfaction": {"dtype": "int64", "null_count": 0, "role": "target"},
        },
        "lookup_tables": {},
        "column_renames": {"Rating_Accuracy ": "rating_accuracy_score"},
    }
    contract.update(overrides)
    return contract


def test_contract_column_renames_are_applied_on_read(tmp_path: Path) -> None:
    """A declared raw-label rename lands the frame on the contract's own names."""
    import pandas as pd

    from broadway.config.schema import DatasetContract
    from broadway.contracts.pandera import build_raw_schema
    from broadway.data.loader import load_with_audit, read_sample

    source = tmp_path / "survey.csv"
    source.write_text("Rating_Accuracy ,satisfaction\n1,2\n5,4\n", encoding="utf-8")
    contract = DatasetContract(**_rename_contract(path=str(source)))

    frame, _, _ = load_with_audit(contract)
    assert list(frame.columns) == ["rating_accuracy_score", "satisfaction"]
    build_raw_schema(contract).validate(frame)

    # A projection names contract columns; the reader maps them back to the file's
    # raw labels (the parquet here keeps the source labels, as a real file does).
    parquet = tmp_path / "survey.parquet"
    pd.read_csv(source).to_parquet(parquet, index=False)
    pruned = read_sample(DatasetContract(**_rename_contract(path=str(parquet))),
                         sample=1, seed=0, columns=["rating_accuracy_score"])
    assert list(pruned.columns) == ["rating_accuracy_score"]

    with pytest.raises(ValueError, match="unbound target"):
        DatasetContract(**_rename_contract(column_renames={"Rating_Accuracy ": "age"}))


def test_dataset_processed_subdir_scopes_canonical_artifacts(tmp_path: Path) -> None:
    """A dataset's processed override outranks the environment default."""
    from broadway.config.schema import DatasetContract, EnvironmentConfig
    from broadway.data.loader import ProcessedStore, canonical_path

    environment = EnvironmentConfig(**{**_ENVIRONMENT, "data_dir": str(tmp_path)})
    scoped = DatasetContract(**_rename_contract(processed_subdir="processed/amazon"))
    shared = DatasetContract(**_rename_contract())

    assert canonical_path(scoped, environment) == (
        tmp_path / "processed" / "amazon" / "survey_canonical.parquet"
    )
    assert ProcessedStore.from_parts(shared, environment).directory == tmp_path / "processed"


def test_artifact_store_scopes_a_dataset_namespace(tmp_path: Path) -> None:
    """A dataset's artifact override outranks the shared dir; a bad one fails at parse."""
    from broadway.config.schema import DatasetContract
    from broadway.data.loader import ArtifactStore

    root = tmp_path / "artifacts" / "discover"
    scoped = DatasetContract(**_rename_contract(artifacts_subdir="amazon"))
    shared = DatasetContract(**_rename_contract())

    assert ArtifactStore.from_contract(scoped, root).directory == root / "amazon"
    assert ArtifactStore.from_contract(shared, root).directory == root
    with pytest.raises(ValueError, match="relative path"):
        DatasetContract(**_rename_contract(artifacts_subdir="../escape"))
