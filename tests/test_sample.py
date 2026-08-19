from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from broadway.config import loader
from broadway.config.loader import load_config
from broadway.lineage import records
from broadway.lineage.models import SampleSpec
from broadway.lineage.sample import load_sample
from broadway.stats import describe as describe_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sample_spec_validation() -> None:
    spec = SampleSpec(name="test", role="estimation", path="data/x.parquet")
    assert spec.name == "test"
    assert spec.role == "estimation"
    assert spec.description is None
    assert spec.column_mapping == {}

    with pytest.raises(ValidationError):
        SampleSpec(name="test", role="bogus", path="data/x.parquet")
    with pytest.raises(ValidationError):
        SampleSpec(name="test", role="diagnostic")
    with pytest.raises(ValidationError):
        SampleSpec(role="diagnostic", path="data/x.parquet")


def test_load_sample_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "foo.yaml").write_text(
        "name: foo\n"
        "role: diagnostic\n"
        "path: results/foo.parquet\n"
        "description: a diagnostic sample\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", tmp_path)

    spec = load_sample("foo")

    assert spec.name == "foo"
    assert spec.role == "diagnostic"
    assert spec.path == "results/foo.parquet"
    assert spec.description == "a diagnostic sample"


def test_load_sample_column_mapping_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "foo.yaml").write_text(
        "name: foo\n"
        "role: diagnostic\n"
        "path: results/foo.parquet\n"
        "description: a diagnostic sample\n"
        "column_mapping:\n"
        "  district: source_district\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", tmp_path)

    spec = load_sample("foo")

    assert spec.column_mapping == {"district": "source_district"}


def _setup_test_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs" / "environment", configs_dir / "environment")
    shutil.copytree(REPO_ROOT / "configs" / "dataset", configs_dir / "dataset")
    shutil.copytree(REPO_ROOT / "configs" / "step", configs_dir / "step")
    (configs_dir / "analysis").mkdir(parents=True)
    (configs_dir / "analysis" / "sample_hypothesis.yaml").write_text(
        "name: sample_hypothesis\n"
        "mode: hypothesis\n"
        "goal: test whether target differs across feature groups\n"
        "row_definition: one listing\n"
        "decision_moment: post-hoc analysis\n"
        "available_info:\n"
        "  - feature_3\n"
        "leakage_notes: []\n"
        "success_criterion: report effect size\n"
        "hypothesis:\n"
        "  group_column: feature_3\n"
        "  group_values:\n"
        "    - A\n"
        "    - B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")


def test_describe_run_stamps_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="sample_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "feature_3": ["A", "A", "B", "B", "B"],
            "target": [10.0, 12.0, 20.0, 22.0, 24.0],
        }
    ).to_parquet(sample_path, index=False)

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )
    sample = SampleSpec(
        name="test_estimation",
        role="estimation",
        path=str(sample_path),
        description="canonical sample",
    )

    describe_module.run(cfg, sample)

    summary = json.loads((tmp_path / "describe.json").read_text())
    assert summary["sample_name"] == "test_estimation"
    assert summary["sample_role"] == "estimation"
    assert summary["source_path"] == str(sample_path)


def test_describe_lineage_sidecar_stamps_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="sample_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "feature_3": ["A", "A", "B", "B", "B"],
            "target": [10.0, 12.0, 20.0, 22.0, 24.0],
        }
    ).to_parquet(sample_path, index=False)

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )
    sample = SampleSpec(
        name="test_estimation",
        role="estimation",
        path=str(sample_path),
    )

    describe_module.run(cfg, sample)

    record_path = tmp_path / "lineage" / "records" / "describe_sample_hypothesis.json"
    record = json.loads(record_path.read_text())
    assert record["sample_name"] == "test_estimation"
    assert record["sample_role"] == "estimation"
