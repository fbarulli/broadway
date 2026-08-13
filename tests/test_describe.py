from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from broadway.config import loader
from broadway.config.loader import load_config
from broadway.lineage import records
from broadway.stats import describe as describe_module
from broadway.stats.describe import (
    GroupSummary,
    describe,
    plot_group_distribution,
    plot_group_sizes,
    run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_describe_groups_and_absent() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "c", "c"], "t": [1, 2, 10, 3, 4]})
    summary = describe(df, "g", ["a", "b", "d"], "t", "path")
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


def test_describe_imbalance_ratio_present_only() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "t": [1, 2, 3, 4, 5]})
    summary = describe(df, "g", ["a", "b", "z"], "t", "path")
    assert summary.groups["z"].n == 0
    assert summary.imbalance_ratio == 1.5


def test_plot_functions_write_files(tmp_path: Path) -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "t": [1.0, 2.0, 3.0]})
    summary = describe(df, "g", ["a", "b"], "t", "path")
    box_path = tmp_path / "box.png"
    sizes_path = tmp_path / "sizes.png"
    plot_group_distribution(df, "g", ["a", "b"], "t", box_path)
    plot_group_sizes(summary, sizes_path)
    assert box_path.exists()
    assert sizes_path.exists()


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
    monkeypatch.setattr(
        describe_module, "canonical_path", lambda dataset, environment: canonical
    )

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    run(cfg)

    assert (tmp_path / "describe.json").exists()
    assert (tmp_path / "describe_boxplot.png").exists()
    assert (tmp_path / "describe_group_sizes.png").exists()


def test_describe_run_missing_canonical_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_test_cfg(tmp_path, monkeypatch)
    cfg = load_config("stats", dataset="test", analysis="test_hypothesis")
    assert cfg.stats is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )
    missing = tmp_path / "does_not_exist_canonical.parquet"
    monkeypatch.setattr(
        describe_module, "canonical_path", lambda dataset, environment: missing
    )

    with pytest.raises(FileNotFoundError):
        run(cfg)
