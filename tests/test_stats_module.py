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


def test_stats_run_absent_declared_group_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Declared [A, B, C] with C absent from data must raise the standard
    named ValueError and write no plan (pre-fix: silent k-1 ANOVA plan)."""
    cfg = _stats_cfg(tmp_path)
    hypothesis = cfg.analysis.hypothesis.model_copy(
        update={"group_values": ["A", "B", "C"]}
    )
    cfg = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"hypothesis": hypothesis})}
    )
    group_col = cfg.analysis.hypothesis.group_column
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            group_col: ["A"] * 6 + ["B"] * 6,
            cfg.dataset.target: np.concatenate(
                [rng.normal(10.0, 2.0, 6), rng.normal(14.0, 2.0, 6)]
            ),
        }
    )
    data_path = tmp_path / "missing_c.parquet"
    df.to_parquet(data_path, index=False)
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: data_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    with pytest.raises(ValueError, match="declared groups absent"):
        module.run(cfg, None)
    assert not (tmp_path / cfg.stats.output_file).exists()


def test_stats_run_small_group_threshold_binds_to_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """any_small_group binds to cfg.stats.min_rows_for_sampling, not the
    library-default 30 hardcoded at the run_anova call site pre-fix: n=5 per
    group is 'small' under the default floor (10000 >= n), not small once the
    configured floor drops below n."""
    default_cfg = _stats_cfg(tmp_path)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            default_cfg.analysis.hypothesis.group_column: ["A"] * 5 + ["B"] * 5,
            default_cfg.dataset.target: np.concatenate(
                [rng.normal(10.0, 2.0, 5), rng.normal(15.0, 2.0, 5)]
            ),
        }
    )
    data_path = tmp_path / "five_per_group.parquet"
    df.to_parquet(data_path, index=False)
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: data_path)
    monkeypatch.setattr(records, "LINEAGE_DIR", tmp_path / "lineage")

    assert default_cfg.stats.min_rows_for_sampling >= 10
    module.run(default_cfg, None)
    plan_default = json.loads((tmp_path / default_cfg.stats.output_file).read_text())
    assert plan_default["threshold_context"]["any_small_group"] is True

    relaxed_stats = default_cfg.stats.model_copy(update={"min_rows_for_sampling": 3})
    relaxed_cfg = default_cfg.model_copy(update={"stats": relaxed_stats})
    module.run(relaxed_cfg, None)
    plan_relaxed = json.loads((tmp_path / relaxed_cfg.stats.output_file).read_text())
    assert plan_relaxed["threshold_context"]["any_small_group"] is False


def test_baseline_hypothesis_absent_declared_group_raises() -> None:
    """Cross-surface vocabulary pin (baseline/hypothesis.py): one declared group
    absent raises the same standard ValueError (pre-fix: silent continue);
    total absence keeps its dedicated 'no groups present' raise. Kept beside
    the other declared-group tripwires; exercises broadway.baseline.hypothesis."""
    from broadway.baseline import hypothesis

    df = pd.DataFrame(
        {"g": ["a", "a", "a", "b", "b", "b"], "t": [10.0] * 3 + [20.0] * 3}
    )
    with pytest.raises(ValueError, match="declared groups absent"):
        hypothesis.run(df, "t", "g", ["a", "b", "c"])

    with pytest.raises(ValueError, match="no groups present"):
        hypothesis.run(df, "t", "g", ["x", "y"])


# --- stats step guard rails -------------------------------------------------


def test_stats_requires_dataset_and_stats_config(tmp_path: Path) -> None:
    cfg = _stats_cfg(tmp_path)
    no_dataset = cfg.model_copy(update={"dataset": None})
    with pytest.raises(ValueError, match="stats step requires dataset and stats config"):
        module.run(no_dataset)
    no_stats = cfg.model_copy(update={"stats": None})
    with pytest.raises(ValueError, match="stats step requires dataset and stats config"):
        module.run(no_stats)


def test_stats_hypothesis_block_required(tmp_path: Path) -> None:
    """model_copy bypasses the contract validator, so a mutated config can
    reach the step with hypothesis mode but no hypothesis block — the step
    must still refuse loudly instead of crashing on None.group_column."""
    cfg = _stats_cfg(tmp_path)
    broken = cfg.model_copy(
        update={"analysis": cfg.analysis.model_copy(update={"hypothesis": None})}
    )
    with pytest.raises(ValueError, match="requires a 'hypothesis' block"):
        module.run(broken)


def test_stats_missing_sample_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)
    sample = SampleSpec(
        name="test_diagnostic", role="diagnostic",
        path=str(tmp_path / "gone_sample.parquet"),
    )
    with pytest.raises(FileNotFoundError, match="sample dataset not found"):
        module.run(cfg, sample)


def test_stats_group_column_not_in_data_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _stats_cfg(tmp_path)
    target_col = cfg.dataset.target
    data_path = tmp_path / "target_only.parquet"
    pd.DataFrame({target_col: [1.0, 2.0, 3.0, 4.0]}).to_parquet(data_path, index=False)
    monkeypatch.setattr(module, "canonical_path", lambda dataset, environment: data_path)
    group_col = cfg.analysis.hypothesis.group_column
    with pytest.raises(ValueError, match=f"group column '{group_col}' not found in data"):
        module.run(cfg)
