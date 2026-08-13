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


def test_stats_run_writes_plan_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("stats", dataset="taxi", experiment="taxi", analysis="taxi_hypothesis")
    assert cfg.stats is not None
    assert cfg.dataset is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    group_col = cfg.analysis.hypothesis.group_column
    target_col = cfg.dataset.target

    def _fake_df() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        frames = [
            pd.DataFrame(
                {
                    group_col: [group] * 5,
                    target_col: rng.normal(mean, 2.0, 5),
                }
            )
            for group, mean in zip(cfg.analysis.hypothesis.group_values, (10.0, 15.0, 20.0, 12.0, 18.0))
        ]
        return pd.concat(frames, ignore_index=True)

    canonical_path = tmp_path / "taxi_canonical.parquet"
    _fake_df().to_parquet(canonical_path, index=False)
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
    assert plan.get("analysis_goal") == "test whether trip duration differs across pickup boroughs"


def test_stats_missing_canonical_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("stats", dataset="taxi", experiment="taxi", analysis="taxi_hypothesis")
    assert cfg.stats is not None
    assert cfg.dataset is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    missing = tmp_path / "does_not_exist_canonical.parquet"
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: missing)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    with pytest.raises(FileNotFoundError):
        module.run(cfg)


def test_stats_run_with_sample_stamps_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("stats", dataset="taxi", experiment="taxi", analysis="taxi_hypothesis")
    assert cfg.stats is not None
    assert cfg.dataset is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    group_col = cfg.analysis.hypothesis.group_column
    target_col = cfg.dataset.target

    def _fake_df() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        frames = [
            pd.DataFrame(
                {
                    group_col: [group] * 5,
                    target_col: rng.normal(mean, 2.0, 5),
                }
            )
            for group, mean in zip(cfg.analysis.hypothesis.group_values, (10.0, 15.0, 20.0, 12.0, 18.0))
        ]
        return pd.concat(frames, ignore_index=True)

    sample_path = tmp_path / "taxi_sample.parquet"
    _fake_df().to_parquet(sample_path, index=False)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    sample = SampleSpec(name="taxi_estimation", role="estimation", path=str(sample_path))
    module.run(cfg, sample)

    plan_path = tmp_path / cfg.stats.output_file
    assert plan_path.exists()

    plan = json.loads(plan_path.read_text())
    assert plan.get("sample_name") == "taxi_estimation"
    assert plan.get("sample_role") == "estimation"
