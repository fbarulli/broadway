from __future__ import annotations

import pandas as pd
import pytest
import yaml

import broadway.onboard.module as module
from broadway.lineage import records


def _write_csv(path) -> str:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "price": [100.0, 150.0, 200.0, 175.0, 125.0],
            "rooms": [2, 3, 4, 3, 2],
            "neighborhood": ["a", "b", "a", "c", "b"],
            "built_at": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        }
    )
    csv = str(path / "houses.csv")
    df.to_csv(csv, index=False)
    return csv


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")


def test_init_writes_configs(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    csv = _write_csv(tmp_path)
    module.init(
        csv,
        "houses",
        target="price",
        task="regression",
        datetime_columns=["built_at"],
        ignore_columns=["id"],
        split_column=None,
        mode="prediction",
        goal="predict price",
        row_definition="one house",
        decision_moment="at listing time",
        available_info=["rooms", "neighborhood", "built_at"],
        leakage_notes=[],
        success_criterion="beats baseline",
    )

    assert (tmp_path / "configs" / "dataset" / "houses.yaml").exists()
    assert (tmp_path / "configs" / "analysis" / "houses.yaml").exists()
    assert (tmp_path / "configs" / "experiment" / "houses.yaml").exists()
    assert (tmp_path / "artifacts" / "discover" / "profile.json").exists()
    assert (tmp_path / "lineage" / "records" / "profile_houses.json").exists()

    dataset = yaml.safe_load((tmp_path / "configs" / "dataset" / "houses.yaml").read_text())
    assert dataset["columns"]["id"]["role"] == "ignore"
    assert dataset["columns"]["price"]["role"] == "target"
    assert dataset["columns"]["built_at"]["role"] == "datetime"

    experiment = yaml.safe_load((tmp_path / "configs" / "experiment" / "houses.yaml").read_text())
    assert experiment["split"]["type"] == "random"


def test_init_time_split(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    csv = _write_csv(tmp_path)
    module.init(
        csv,
        "houses",
        target="price",
        task="regression",
        datetime_columns=["built_at"],
        ignore_columns=["id"],
        split_column="built_at",
        mode="prediction",
        goal="predict price",
        row_definition="one house",
        decision_moment="at listing time",
        available_info=["rooms", "neighborhood", "built_at"],
        leakage_notes=[],
        success_criterion="beats baseline",
    )

    dataset = yaml.safe_load((tmp_path / "configs" / "dataset" / "houses.yaml").read_text())
    assert dataset["datetime_column"] == "built_at"

    experiment = yaml.safe_load((tmp_path / "configs" / "experiment" / "houses.yaml").read_text())
    assert experiment["split"]["type"] == "time"


def test_init_non_tty_missing_required_raises(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    csv = _write_csv(tmp_path)

    class _FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stdin", _FakeStdin())
    with pytest.raises(ValueError, match="missing required"):
        module.init(
            csv,
            "houses",
            target=None,
            task=None,
            mode=None,
            goal=None,
            row_definition=None,
            decision_moment=None,
            success_criterion=None,
        )
