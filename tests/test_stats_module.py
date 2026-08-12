from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadway.config.loader import load_config
from broadway.config.schema import DatasetContract, PipelineConfig
from broadway.stats import module


def test_stats_run_writes_plan_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("stats", dataset="taxi", experiment="taxi")
    assert cfg.stats is not None
    assert cfg.dataset is not None

    cfg = cfg.model_copy(
        update={"stats": cfg.stats.model_copy(update={"output_dir": str(tmp_path)})}
    )

    group_col = cfg.stats.group_column
    target_col = cfg.dataset.target

    def _fake_load(dataset: DatasetContract) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        frames = [
            pd.DataFrame(
                {
                    group_col: [group] * 5,
                    target_col: rng.normal(mean, 2.0, 5),
                }
            )
            for group, mean in zip(cfg.stats.group_values, (10.0, 15.0, 20.0, 12.0, 18.0))
        ]
        return pd.concat(frames, ignore_index=True)

    monkeypatch.setattr(module, "load", _fake_load)

    module.run(cfg)

    plan_path = tmp_path / cfg.stats.output_file
    assert plan_path.exists()

    plan = json.loads(plan_path.read_text())
    assert "statistics" in plan
    assert "effect_sizes" in plan
    assert "passed" in plan
    assert "reason" in plan
