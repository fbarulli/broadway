from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import PipelineConfig
from broadway.lineage import records
from broadway.lineage.models import SampleSpec
from broadway.stats import module


def _stats_cfg(tmp_path: Path) -> PipelineConfig:
    cfg = load_config("stats", dataset="test", experiment="baseline", analysis="test_hypothesis")
    assert cfg.stats is not None
    assert cfg.dataset is not None
    return cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )


def _fake_df(cfg: PipelineConfig) -> pd.DataFrame:
    group_col = cfg.analysis.hypothesis.group_column
    target_col = cfg.dataset.target
    rng = np.random.default_rng(42)
    frames = [
        pd.DataFrame(
            {
                group_col: [group] * 5,
                target_col: rng.normal(mean, 2.0, 5),
            }
        )
        for group, mean in zip(cfg.analysis.hypothesis.group_values, (10.0, 15.0))
    ]
    return pd.concat(frames, ignore_index=True)


def test_stats_run_writes_plan_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)

    canonical_path = tmp_path / "test_canonical.parquet"
    _fake_df(cfg).to_parquet(canonical_path, index=False)
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: canonical_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    module.run(cfg)

    plan_path = tmp_path / cfg.stats.output_file
    assert plan_path.exists()

    plan = json.loads(plan_path.read_text())
    assert "statistics" in plan
    assert "effect_sizes" in plan
    assert "passed" in plan
    assert "reason" in plan
    assert plan.get("analysis_goal") == "test whether target differs across feature groups"


def test_stats_missing_canonical_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)

    missing = tmp_path / "does_not_exist_canonical.parquet"
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: missing)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    with pytest.raises(FileNotFoundError):
        module.run(cfg)


def test_stats_run_with_sample_stamps_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)

    sample_path = tmp_path / "test_sample.parquet"
    _fake_df(cfg).to_parquet(sample_path, index=False)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    sample = SampleSpec(name="test_estimation", role="estimation", path=str(sample_path))
    module.run(cfg, sample)

    plan_path = tmp_path / cfg.stats.output_file
    assert plan_path.exists()

    plan = json.loads(plan_path.read_text())
    assert plan.get("sample_name") == "test_estimation"
    assert plan.get("sample_role") == "estimation"


def test_stats_run_with_sample_column_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)

    target_col = cfg.dataset.target
    group_col = cfg.analysis.hypothesis.group_column

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            group_col: ["A"] * 5 + ["B"] * 5,
            target_col: rng.normal(0.0, 2.0, 10),
        }
    )

    sample_path = tmp_path / "test_sample.parquet"
    df.to_parquet(sample_path, index=False)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    sample = SampleSpec(
        name="test_diagnostic",
        role="diagnostic",
        path=str(sample_path),
        column_mapping={group_col: group_col},
    )
    module.run(cfg, sample)

    plan_path = tmp_path / cfg.stats.output_file
    assert plan_path.exists()

    plan = json.loads(plan_path.read_text())
    assert plan.get("sample_name") == "test_diagnostic"
    assert plan.get("sample_role") == "diagnostic"
