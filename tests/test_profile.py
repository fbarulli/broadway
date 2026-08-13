from __future__ import annotations

import pandas as pd
import pytest
import yaml

import broadway.discover.module as module
from broadway.discover.profile import DatasetProfile, build_profile
from broadway.lineage import records


def test_build_profile_fields() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
            "group": ["a", "a", "b", "b"],
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
        }
    )
    profile = build_profile("taxi", "data.csv", df)
    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 4
    assert set(profile.columns) == {"id", "value", "group", "ts"}
    assert profile.columns["value"].cardinality == 4
    assert profile.columns["group"].cardinality == 2
    assert profile.columns["id"].identifier_score == pytest.approx(1.0)
    assert profile.columns["value"].min == "10.0"
    assert profile.columns["value"].max == "40.0"


def test_build_profile_datetime_min_max() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            )
        }
    )
    profile = build_profile("taxi", "data.csv", df)
    ts = profile.columns["ts"]
    assert ts.dtype.startswith("datetime64")
    assert ts.datetime_min is not None
    assert ts.datetime_max is not None


def test_build_profile_null_count() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df.loc[0, "value"] = None
    profile = build_profile("taxi", "data.csv", df)
    assert profile.columns["value"].null_count == 1


def test_run_writes_profile(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_csv(csv, index=False)
    monkeypatch.setattr(module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(module, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(module, "DATASET_DIR", "dataset")
    module.run(str(csv), "value", "regression")
    profile_path = tmp_path / "artifacts" / "discover" / "profile.json"
    assert profile_path.exists()
    profile = DatasetProfile.model_validate_json(profile_path.read_text())
    assert profile.name == "data"


def test_profile_command(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame({"a": [1, 2, 3], "t": [10.0, 20.0, 30.0]})
    parquet = tmp_path / "data.parquet"
    data.to_parquet(parquet)

    monkeypatch.setattr(module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(module, "ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(module, "DATASET_DIR", "dataset")
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    config_dir = tmp_path / "configs" / "dataset"
    config_dir.mkdir(parents=True)
    contract = {
        "name": "mydata",
        "path": str(parquet),
        "target": "t",
        "task": "regression",
        "datetime_column": None,
        "columns": {
            "a": {"dtype": "int64", "null_count": 0, "role": "feature"},
            "t": {"dtype": "float64", "null_count": 0, "role": "target"},
        },
        "lookup_tables": {},
        "row_count": 3,
    }
    (config_dir / "mydata.yaml").write_text(
        yaml.safe_dump(contract), encoding="utf-8"
    )

    module.profile("mydata")

    assert (tmp_path / "artifacts" / "discover" / "profile.json").exists()
    assert (tmp_path / "lineage" / "records" / "profile_mydata.json").exists()
