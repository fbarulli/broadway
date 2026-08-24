"""Recipe builder + schema-contract cross-check tests — synthetic data only.

Raw + lookup CSVs are generated under ``tmp_path``; nothing reads real data
(enforced by tests/test_platform_hygiene.py). The lookup fixture mirrors
taxi's join shape: two lookups keyed on columns absent from the raw contract,
so the second merge collides and produces ``*_lookup`` names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from broadway.config import loader as config_loader
from broadway.config.loader import load_config
from broadway.config.schema import (
    ColumnRole,
    ColumnSchema,
    DatasetContract,
    LookupSpec,
    PreprocessingStepConfig,
    TaskType,
)
from broadway.data.loader import load_with_audit
from broadway.features.recipe import build_pipeline, validate_preprocessing_columns
from broadway.schemas import schema_columns
from broadway.schemas.joined import joined_schema_columns


@pytest.fixture
def lookup_dataset(tmp_path: Path) -> DatasetContract:
    """Synthetic lookup-bearing dataset mirroring taxi's join shape (two lookups)."""
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "pickup_location_id": [1, 2, 3],
            "dropoff_location_id": [2, 3, 1],
            "fare": [10.0, 20.0, 30.0],
            "duration": [100, 200, 300],
        }
    ).to_csv(raw_path, index=False)

    zones1_path = tmp_path / "zones1.csv"
    pd.DataFrame(
        {"LocationID": [1, 2, 3], "Borough": ["A", "B", "C"], "Zone": ["Z1", "Z2", "Z3"]}
    ).to_csv(zones1_path, index=False)

    zones2_path = tmp_path / "zones2.csv"
    pd.DataFrame(
        {
            "LocationID": [1, 2, 3],
            "Borough": ["X", "Y", "Z"],
            "service_zone": ["S1", "S2", "S3"],
        }
    ).to_csv(zones2_path, index=False)

    return DatasetContract(
        name="synthetic",
        path=str(raw_path),
        target="duration",
        task=TaskType.REGRESSION,
        datetime_column=None,
        columns={
            "pickup_location_id": ColumnSchema(
                dtype="int64", null_count=0, role=ColumnRole.FEATURE
            ),
            "dropoff_location_id": ColumnSchema(
                dtype="int64", null_count=0, role=ColumnRole.FEATURE
            ),
            "fare": ColumnSchema(dtype="float64", null_count=0, role=ColumnRole.FEATURE),
            "duration": ColumnSchema(dtype="int64", null_count=0, role=ColumnRole.TARGET),
        },
        lookup_tables={
            "pickup_location_id": LookupSpec(path=str(zones1_path), key="LocationID"),
            "dropoff_location_id": LookupSpec(path=str(zones2_path), key="LocationID"),
        },
    )


def _recipe_cfg(
    preprocessing: list[dict[str, object]],
    dataset: DatasetContract,
    schema_contract: str,
) -> object:
    cfg = load_config("train", dataset="test", experiment="baseline")
    experiment = cfg.experiment.model_copy(
        update={
            "data_source": cfg.experiment.data_source.model_copy(
                update={"schema_contract": schema_contract}
            ),
            "preprocessing": [PreprocessingStepConfig(**step) for step in preprocessing],
        }
    )
    return cfg.model_copy(update={"dataset": dataset, "experiment": experiment})


def test_empty_preprocessing_is_passthrough_identity() -> None:
    cfg = load_config("train", dataset="test", experiment="baseline")
    pipeline = build_pipeline(cfg)
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["passthrough"]
    X = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    pd.testing.assert_frame_equal(pipeline.transform(X), X)


def test_yaml_to_pipeline_round_trip(lookup_dataset: DatasetContract) -> None:
    steps = yaml.safe_load(
        """
        - type: target_encoding
          columns: [pickup_location_id]
          params: {smoothing: 20}
        - type: frequency_encoding
          columns: [dropoff_location_id]
          params: {normalize: false}
        - type: passthrough
          columns: [fare]
        """
    )
    cfg = _recipe_cfg(steps, lookup_dataset, "joined")
    pipeline = build_pipeline(cfg)
    assert [name for name, _ in pipeline.steps] == [
        "target_encoding_0",
        "frequency_encoding_1",
        "passthrough_2",
    ]
    params = pipeline.get_params()
    assert params["target_encoding_0__smoothing"] == 20
    assert params["frequency_encoding_1__normalize"] is False


def test_one_hot_sole_step_builds(lookup_dataset: DatasetContract) -> None:
    cfg = _recipe_cfg([{"type": "one_hot", "columns": ["Borough"]}], lookup_dataset, "joined")
    pipeline = build_pipeline(cfg)
    assert [name for name, _ in pipeline.steps] == ["one_hot_0"]
    assert pipeline.get_params()["one_hot_0__one_hot__handle_unknown"] == "ignore"


def test_unknown_step_type_rejected_at_parse() -> None:
    with pytest.raises(ValidationError):
        PreprocessingStepConfig(type="bogus", columns=["fare"])


def test_invalid_step_param_fails_loud(lookup_dataset: DatasetContract) -> None:
    cfg = _recipe_cfg(
        [
            {
                "type": "target_encoding",
                "columns": ["pickup_location_id"],
                "params": {"bogus": 1},
            }
        ],
        lookup_dataset,
        "joined",
    )
    with pytest.raises(ValueError, match="bogus"):
        build_pipeline(cfg)


def test_target_encoding_requires_smoothing(lookup_dataset: DatasetContract) -> None:
    cfg = _recipe_cfg(
        [{"type": "target_encoding", "columns": ["pickup_location_id"]}],
        lookup_dataset,
        "joined",
    )
    with pytest.raises(ValueError, match="requires param"):
        build_pipeline(cfg)


def test_cross_check_rejects_column_outside_bound_schema(
    lookup_dataset: DatasetContract,
) -> None:
    cfg = _recipe_cfg(
        [{"type": "one_hot", "columns": ["nonexistent_col"]}], lookup_dataset, "joined"
    )
    with pytest.raises(ValueError) as exc_info:
        validate_preprocessing_columns(cfg)
    message = str(exc_info.value)
    assert "one_hot" in message
    assert "nonexistent_col" in message
    assert "joined" in message


def test_lookup_derived_column_passes_joined_fails_raw(
    lookup_dataset: DatasetContract,
) -> None:
    steps = [{"type": "one_hot", "columns": ["Borough"]}]
    validate_preprocessing_columns(_recipe_cfg(steps, lookup_dataset, "joined"))
    with pytest.raises(ValueError, match="Borough"):
        validate_preprocessing_columns(_recipe_cfg(steps, lookup_dataset, "raw"))


def test_joined_schema_matches_loader_output(lookup_dataset: DatasetContract) -> None:
    """Equivalence pin (Decision 6): the joined schema's column set is exactly
    what ``load_with_audit`` actually emits — no second copy of the suffix rule."""
    emitted = set(load_with_audit(lookup_dataset)[0].columns)
    assert joined_schema_columns(lookup_dataset) == frozenset(emitted)
    assert "Borough" in emitted
    assert "Borough_lookup" in emitted
    assert "LocationID" in emitted
    assert "LocationID_lookup" in emitted
    assert "service_zone" in emitted


def test_unknown_schema_contract_fails_loud(lookup_dataset: DatasetContract) -> None:
    with pytest.raises(ValueError, match="unknown schema_contract 'bogus'"):
        schema_columns("bogus", lookup_dataset)


def _write_config_tree(tmp_path: Path, preprocessing: list[dict[str, object]]) -> None:
    environment = {
        "log_level": "INFO",
        "data_dir": "data",
        "raw_subdir": "raw",
        "processed_subdir": "processed",
        "download_chunk_size": 8192,
        "mlflow_tracking_uri": "mlruns",
        "database_user": "user",
        "database_password": "pass",
        "database_name": "db",
        "database_host": "localhost",
        "database_port": 5432,
        "sample_size_ci": 1000,
        "sample_size_stats": 10000,
        "api_replicas_min": 1,
        "api_replicas_max": 3,
        "api_hpa_cpu_threshold": 80,
    }
    dataset = {
        "name": "syn",
        "path": "syn.parquet",
        "target": "price",
        "task": "regression",
        "datetime_column": None,
        "columns": {
            "a": {"dtype": "float64", "null_count": 0, "role": "feature"},
            "price": {"dtype": "float64", "null_count": 0, "role": "target"},
        },
        "lookup_tables": {},
    }
    experiment = {
        "data_source": {"loader": "canonical", "schema_contract": "raw"},
        "features": {"include": ["a"], "exclude": [], "derived": [], "encodings": []},
        "model": {"type": "linear", "params": {}},
        "split": {"type": "random", "validation_size": 0.2},
        "random_state": 42,
        "target_metric": "rmse",
        "preprocessing": preprocessing,
    }
    files = {
        "environment/development.yaml": environment,
        "dataset/syn.yaml": dataset,
        "experiment/syn.yaml": experiment,
        "step/baseline.yaml": {"output_dir": "artifacts/baseline", "output_file": "baseline.json"},
    }
    for relative, payload in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_cross_check_fires_at_config_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config_tree(tmp_path, [{"type": "one_hot", "columns": ["bogus"]}])
    monkeypatch.setattr(config_loader, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="bogus"):
        load_config("baseline", dataset="syn", experiment="syn")


def test_config_load_accepts_valid_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config_tree(tmp_path, [{"type": "one_hot", "columns": ["a"]}])
    monkeypatch.setattr(config_loader, "CONFIGS_DIR", tmp_path)
    cfg = load_config("baseline", dataset="syn", experiment="syn")
    assert cfg.experiment is not None
    assert cfg.experiment.preprocessing[0].columns == ["a"]


def test_one_hot_mixed_with_other_steps_rejected_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config_tree(
        tmp_path,
        [
            {"type": "one_hot", "columns": ["a"]},
            {"type": "frequency_encoding", "columns": ["a"]},
        ],
    )
    monkeypatch.setattr(config_loader, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="one_hot must be the sole preprocessing step"):
        load_config("baseline", dataset="syn", experiment="syn")


def test_normalize_accepts_string_bools(lookup_dataset: DatasetContract) -> None:
    for raw, expected in (("true", True), ("false", False), ("TRUE", True), ("False", False)):
        cfg = _recipe_cfg(
            [
                {
                    "type": "frequency_encoding",
                    "columns": ["pickup_location_id"],
                    "params": {"normalize": raw},
                }
            ],
            lookup_dataset,
            "joined",
        )
        pipeline = build_pipeline(cfg)
        assert pipeline.get_params()["frequency_encoding_0__normalize"] is expected


def test_normalize_rejects_non_bool_values(lookup_dataset: DatasetContract) -> None:
    cfg = _recipe_cfg(
        [
            {
                "type": "frequency_encoding",
                "columns": ["pickup_location_id"],
                "params": {"normalize": 1},
            }
        ],
        lookup_dataset,
        "joined",
    )
    with pytest.raises(ValueError, match="normalize"):
        build_pipeline(cfg)
