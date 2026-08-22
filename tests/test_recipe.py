"""Recipe builder + schema-contract cross-check tests — synthetic data only.

Mirrors ``test_structural_cleaning.py::test_etl_with_lookups_writes_join_audit``:
raw + lookup CSVs are generated under ``tmp_path``; nothing reads real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

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
        - type: one_hot
          columns: [Borough]
        - type: passthrough
          columns: [fare]
        """
    )
    cfg = _recipe_cfg(steps, lookup_dataset, "joined")
    pipeline = build_pipeline(cfg)
    assert [name for name, _ in pipeline.steps] == [
        "target_encoding_0",
        "frequency_encoding_1",
        "one_hot_2",
        "passthrough_3",
    ]
    params = pipeline.get_params()
    assert params["target_encoding_0__smoothing"] == 20
    assert params["frequency_encoding_1__normalize"] is False
    assert params["one_hot_2__one_hot__handle_unknown"] == "ignore"


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
    emitted = set(load_with_audit(lookup_dataset)[0].columns)
    assert joined_schema_columns(lookup_dataset) == frozenset(emitted)
    assert "Borough_lookup" in emitted
    assert "LocationID_lookup" in emitted


def test_unknown_schema_contract_fails_loud(lookup_dataset: DatasetContract) -> None:
    with pytest.raises(ValueError, match="raw, joined"):
        schema_columns("bogus", lookup_dataset)
