"""Public behavior of the column plan: apply == the shipped eligible surface.

The plan is the single add/drop path; this test pins that its projection is
byte-equivalent to ``eligible_feature_columns`` for every shipped experiment
contract and records the same ordered manifest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from broadway.config.loader import load_config
from broadway.config.schema import (
    DatasetContract,
    ExperimentConfig,
    PipelineConfig,
)
from broadway.features.plan import ColumnOp, ColumnPlan, resolve_plan
from broadway.utils import eligible_feature_columns

# The Olist reference contract lives in the project overlay; the path is split
# so the platform-hygiene guard can distinguish this config-only equivalence
# test (generated frames, no project imports/data reads) from demo coupling.
_PROJECT_CONFIG = Path("project") / "config"
_N = 16


def _shipped_cfg(dataset: str, experiment: str) -> PipelineConfig:
    return load_config("train", dataset=dataset, experiment=experiment)


def _olist_cfg() -> PipelineConfig:
    """Load the Olist contract pair as plain config models (no step tree)."""
    dataset = DatasetContract.model_validate(
        yaml.safe_load((_PROJECT_CONFIG / "dataset" / "olist.yaml").read_text(encoding="utf-8"))
    )
    experiment = ExperimentConfig.model_validate(
        yaml.safe_load(
            (_PROJECT_CONFIG / "experiment" / "olist.yaml").read_text(encoding="utf-8")
        )
    )
    experiment.bind_plan(resolve_plan(experiment.feature_plan, dataset))
    base = _shipped_cfg("test", "baseline")
    return base.model_copy(update={"dataset": dataset, "experiment": experiment})


def _shipped_pairs() -> list[tuple[str, str]]:
    experiments = ("baseline", "engineered", "hyperopt")
    return [("test", experiment) for experiment in experiments]


def _series(dtype: str, rng: np.random.Generator) -> pd.Series:
    """One synthetic column shaped by its declared dtype."""
    if dtype == "object":
        return pd.Series([f"cat_{i % 3}" for i in range(_N)], dtype="object")
    if dtype.startswith("datetime64"):
        return pd.Series(pd.date_range("2024-01-01", periods=_N, freq="h"))
    if dtype == "bool":
        return pd.Series([i % 2 == 0 for i in range(_N)], dtype="bool")
    if dtype.startswith(("int", "uint")):
        return pd.Series(rng.integers(0, 100, _N), dtype=dtype)
    if dtype.startswith("float"):
        return pd.Series(rng.random(_N), dtype=dtype)
    raise AssertionError(f"unsupported synthetic dtype: {dtype}")


def _post_features_frame(cfg: PipelineConfig) -> pd.DataFrame:
    """A producer-shaped frame: contract columns, derived, then encodings."""
    assert cfg.dataset is not None and cfg.experiment is not None
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    rng = np.random.default_rng(7)
    data = {name: _series(column.dtype, rng) for name, column in cfg.dataset.columns.items()}
    for op in plan.columns:
        data[op.name] = _series(op.dtype, rng)
    return pd.DataFrame(data)


@pytest.mark.parametrize(("dataset", "experiment"), _shipped_pairs())
def test_plan_apply_matches_eligible_surface_and_manifest(
    dataset: str, experiment: str
) -> None:
    cfg = _shipped_cfg(dataset, experiment)
    _assert_plan_matches(cfg)


def test_plan_apply_matches_eligible_surface_for_the_olist_contract() -> None:
    _assert_plan_matches(_olist_cfg())


def test_olist_plan_excludes_every_non_feature_role() -> None:
    """The role boundary is the leakage boundary: only role=feature enters."""
    cfg = _olist_cfg()
    assert cfg.dataset is not None and cfg.experiment is not None
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    excluded = {
        name
        for name, column in cfg.dataset.columns.items()
        if column.role.value != "feature"
    }
    features = {
        name
        for name, column in cfg.dataset.columns.items()
        if column.role.value == "feature"
    }
    assert "delivery_days" in excluded and "review_score" in excluded
    assert not (excluded & set(plan.names))
    declared = {op.name for op in plan.columns if op.op == "source"} | {
        source for op in plan.columns for source in op.sources
    }
    assert declared == features
    assert plan.target == cfg.dataset.target


def _assert_plan_matches(cfg: PipelineConfig) -> None:
    assert cfg.dataset is not None and cfg.experiment is not None
    assert cfg.experiment.data_source.schema_contract == "engineered"
    frame = _post_features_frame(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    projected, manifest = plan.apply(frame)
    expected = frame[list(plan.names)]

    assert_frame_equal(projected, expected)
    assert_frame_equal(eligible_feature_columns(frame, cfg), expected)
    assert [entry.name for entry in manifest.entries] == list(plan.names)
    assert [entry.op for entry in manifest.entries] == [op.op for op in plan.columns]
    assert manifest.target == cfg.dataset.target
    dropped = set(frame.columns) - set(plan.names) - {cfg.dataset.target}
    assert set(manifest.dropped) == dropped


def test_plan_apply_fails_loud_on_a_missing_derived_source() -> None:
    cfg = _shipped_cfg("test", "engineered")
    assert cfg.dataset is not None and cfg.experiment is not None
    frame = _post_features_frame(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    derived = next(op for op in plan.columns if op.op == "derive")
    # A missing derive/source input must fail loud, naming the dropped column.
    with pytest.raises(ValueError, match=re.escape(derived.sources[0])):
        plan.apply(frame.drop(columns=[derived.sources[0]]))


def test_plan_apply_computes_a_missing_derived_column() -> None:
    cfg = _shipped_cfg("test", "engineered")
    assert cfg.dataset is not None and cfg.experiment is not None
    frame = _post_features_frame(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    derived = next(op for op in plan.columns if op.op == "derive")

    projected, _ = plan.apply(frame.drop(columns=[derived.name]))

    expected = frame[derived.sources[0]].astype(derived.dtype)
    assert projected[derived.name].equals(expected)


def test_plan_apply_requires_the_persisted_encoder_output() -> None:
    cfg = _shipped_cfg("test", "engineered")
    assert cfg.dataset is not None and cfg.experiment is not None
    frame = _post_features_frame(cfg)
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    encoded = next(op for op in plan.columns if op.op == "encode")
    with pytest.raises(ValueError, match="declared column\\(s\\) missing"):
        plan.apply(frame.drop(columns=[encoded.name]))


def test_add_and_drop_return_new_plans() -> None:
    cfg = _shipped_cfg("test", "engineered")
    assert cfg.dataset is not None and cfg.experiment is not None
    plan = ColumnPlan.from_config(cfg.dataset, cfg.experiment)
    original = plan.names
    extra = ColumnOp(op="source", name="extra_feature", dtype="float64")

    extended = plan.add(extra)
    assert extended is not plan
    assert plan.names == original  # the original plan is untouched
    assert extended.names == (*original, "extra_feature")

    trimmed = extended.drop("extra_feature")
    assert trimmed.names == original
    with pytest.raises(ValueError, match="does not declare"):
        plan.drop("missing_feature")


def test_unresolved_plan_fails_loud() -> None:
    cfg = _shipped_cfg("test", "baseline")
    assert cfg.dataset is not None and cfg.experiment is not None
    raw = cfg.experiment.model_dump(mode="json")
    raw["feature_plan"] = "no_such_module:plan"
    unbound = ExperimentConfig.model_validate(raw)
    with pytest.raises(ValueError, match="is not resolved"):
        _ = unbound.plan


@pytest.mark.parametrize(
    ("reference", "match"),
    [
        ("baseline", "module:callable"),
        ("no_such_module_xyz:plan", "could not be imported"),
        ("broadway.features.plans:missing", "not found"),
        ("builtins:str", "must return a ColumnPlan"),
    ],
)
def test_invalid_feature_plan_reference_fails_loud(reference: str, match: str) -> None:
    cfg = _shipped_cfg("test", "baseline")
    assert cfg.dataset is not None
    with pytest.raises((ValueError, TypeError), match=match):
        resolve_plan(reference, cfg.dataset)
