from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from broadway.config import loader
from broadway.config.loader import load_config
from broadway.lineage import records
from broadway.lineage.models import SampleSpec
from broadway.reports import paths
from broadway.stats.describe import (
    GroupSummary,
    describe,
    plot_describe_figures,
    run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_describe_groups_and_absent() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "c", "c"], "t": [1, 2, 10, 3, 4]})
    summary = describe(df, "g", "g", ["a", "b", "d"], "t", "path", "s", "diagnostic")
    assert summary.total_n == 5
    assert summary.groups["a"].n == 2
    assert summary.groups["b"].n == 1
    assert "d" in summary.absent_groups
    assert summary.groups["d"].n == 0
    assert summary.groups["d"].mean is None
    assert summary.imbalance_ratio == 2.0
    assert set(summary.proportions) == {"a", "b", "d"}


def test_describe_no_verdict() -> None:
    for banned in ("balanced", "unbalanced", "verdict"):
        assert banned not in GroupSummary.model_fields
    assert "canonical_path" not in GroupSummary.model_fields
    for field in ("sample_name", "sample_role", "source_path"):
        assert field in GroupSummary.model_fields


def test_describe_imbalance_ratio_present_only() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "t": [1, 2, 3, 4, 5]})
    summary = describe(df, "g", "g", ["a", "b", "z"], "t", "path", "s", "diagnostic")
    assert summary.groups["z"].n == 0
    assert summary.imbalance_ratio == 1.5


def test_plot_functions_write_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "t": [1.0, 2.0, 3.0]})
    summary = describe(df, "g", "g", ["a", "b"], "t", "path", "s", "diagnostic")
    out = tmp_path / "describe.png"
    plot_describe_figures(df, "g", "g", ["a", "b"], "t", summary, out)
    assert out.exists()


def _setup_test_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs" / "environment", configs_dir / "environment")
    shutil.copytree(REPO_ROOT / "configs" / "dataset", configs_dir / "dataset")
    shutil.copytree(REPO_ROOT / "configs" / "step", configs_dir / "step")
    (configs_dir / "analysis").mkdir(parents=True)
    (configs_dir / "analysis" / "test_hypothesis.yaml").write_text(
        "name: test_hypothesis\n"
        "mode: hypothesis\n"
        "goal: test whether price differs across neighborhoods\n"
        "row_definition: one listing\n"
        "decision_moment: post-hoc\n"
        "available_info:\n"
        "  - neighborhood\n"
        "leakage_notes: []\n"
        "success_criterion: report effect size\n"
        "hypothesis:\n"
        "  group_column: neighborhood\n"
        "  group_values:\n"
        "    - A\n"
        "    - B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "reports" / "results")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "reports" / "figures")


def test_describe_run_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="test_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    canonical = tmp_path / "test_canonical.parquet"
    pd.DataFrame(
        {"neighborhood": ["A", "A", "B", "B", "B"], "price": [100, 110, 200, 210, 220]}
    ).to_parquet(canonical, index=False)
    sample = SampleSpec(name="test_diagnostic", role="diagnostic", path=str(canonical))

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    run(cfg, sample)

    assert (tmp_path / "describe.json").exists()
    assert (tmp_path / "lineage" / "records" / "describe_test_hypothesis.json").exists()
    assert not (tmp_path / "reports" / "results" / "describe.md").exists()
    assert not (tmp_path / "reports" / "figures" / "describe.png").exists()


def test_describe_run_missing_sample_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="test_hypothesis")
    assert cfg.stats is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )
    missing = tmp_path / "does_not_exist_sample.parquet"
    sample = SampleSpec(name="test_diagnostic", role="diagnostic", path=str(missing))

    with pytest.raises(FileNotFoundError):
        run(cfg, sample)


def test_describe_run_column_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs" / "environment", configs_dir / "environment")
    shutil.copytree(REPO_ROOT / "configs" / "dataset", configs_dir / "dataset")
    shutil.copytree(REPO_ROOT / "configs" / "step", configs_dir / "step")
    (configs_dir / "analysis").mkdir(parents=True)
    (configs_dir / "analysis" / "borough_hypothesis.yaml").write_text(
        "name: borough_hypothesis\n"
        "mode: hypothesis\n"
        "goal: test whether price differs across boroughs\n"
        "row_definition: one listing\n"
        "decision_moment: post-hoc\n"
        "available_info:\n"
        "  - Borough\n"
        "leakage_notes: []\n"
        "success_criterion: report effect size\n"
        "hypothesis:\n"
        "  group_column: Borough\n"
        "  group_values:\n"
        "    - Manhattan\n"
        "    - Brooklyn\n"
        "    - Queens\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "reports" / "results")
    monkeypatch.setattr(paths, "FIGURES_DIR", tmp_path / "reports" / "figures")

    cfg = load_config("stats", dataset="test", analysis="borough_hypothesis")
    assert cfg.dataset is not None
    assert cfg.stats is not None

    sample_path = tmp_path / "mapped_sample.parquet"
    pd.DataFrame(
        {
            "pickup_borough": ["Manhattan", "Manhattan", "Brooklyn", "Brooklyn", "Brooklyn"],
            "price": [100, 110, 200, 210, 220],
        }
    ).to_parquet(sample_path, index=False)

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )
    sample = SampleSpec(
        name="taxi_diagnostic",
        role="diagnostic",
        path=str(sample_path),
        description="mapped sample",
        column_mapping={"Borough": "pickup_borough"},
    )

    run(cfg, sample)

    summary = json.loads((tmp_path / "describe.json").read_text())
    assert summary["group_column"] == "Borough"
    assert summary["source_group_column"] == "pickup_borough"
    assert set(summary["groups"]) == {"Manhattan", "Brooklyn", "Queens"}
    assert summary["groups"]["Manhattan"]["n"] == 2
    assert summary["groups"]["Brooklyn"]["n"] == 3
    assert summary["groups"]["Queens"]["n"] == 0
    assert "Queens" in summary["absent_groups"]
