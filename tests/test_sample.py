from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from contract_fixture import categorical_column, target_column
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


def test_load_sample_prefers_project_overlay_and_falls_back_to_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_sample_dir = tmp_path / "base" / "sample"
    overlay_sample_dir = tmp_path / "overlay" / "sample"
    base_sample_dir.mkdir(parents=True)
    overlay_sample_dir.mkdir(parents=True)
    (base_sample_dir / "foo.yaml").write_text(
        "name: foo\nrole: diagnostic\npath: results/base-foo.parquet\n", encoding="utf-8"
    )
    (base_sample_dir / "base_only.yaml").write_text(
        "name: base_only\nrole: diagnostic\npath: results/base-only.parquet\n", encoding="utf-8"
    )
    (overlay_sample_dir / "foo.yaml").write_text(
        "name: foo\nrole: diagnostic\npath: results/overlay-foo.parquet\n", encoding="utf-8"
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", tmp_path / "base")
    monkeypatch.setenv("BROADWAY_CONFIG_OVERLAY_DIR", str(tmp_path / "overlay"))

    assert load_sample("foo").path == "results/overlay-foo.parquet"
    assert load_sample("base_only").path == "results/base-only.parquet"


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


def _demo_columns() -> tuple[str, str]:
    """(group column, target column) from the demo dataset contract."""
    cfg = load_config("contracts", dataset="test", experiment="baseline")
    assert cfg.dataset is not None
    group = categorical_column(cfg.dataset)
    assert group is not None
    return group, target_column(cfg.dataset)


def _setup_test_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group, _ = _demo_columns()
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
        f"available_info:\n"
        f"  - {group}\n"
        "leakage_notes: []\n"
        "success_criterion: report effect size\n"
        "hypothesis:\n"
        f"  group_column: {group}\n"
        "  group_values:\n"
        "    - A\n"
        "    - B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")


def _mapped_sample(tmp_path: Path, group: str, target: str) -> SampleSpec:
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            group: ["A", "A", "B", "B", "B"],
            target: [10.0, 12.0, 20.0, 22.0, 24.0],
        }
    ).to_parquet(sample_path, index=False)
    return SampleSpec(
        name="test_estimation",
        role="estimation",
        path=str(sample_path),
        description="canonical sample",
    )


def test_describe_run_stamps_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group, target = _demo_columns()
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="sample_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    sample = _mapped_sample(tmp_path, group, target)

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    describe_module.run(cfg, sample)

    summary = json.loads((tmp_path / "describe.json").read_text())
    assert summary["sample_name"] == "test_estimation"
    assert summary["sample_role"] == "estimation"
    assert summary["source_path"] == str(sample.path)


def test_describe_lineage_sidecar_stamps_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group, target = _demo_columns()
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="sample_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    sample = _mapped_sample(tmp_path, group, target)

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    describe_module.run(cfg, sample)

    record_path = tmp_path / "lineage" / "records" / "describe_sample_hypothesis.json"
    record = json.loads(record_path.read_text())
    assert record["sample_name"] == "test_estimation"
    assert record["sample_role"] == "estimation"
