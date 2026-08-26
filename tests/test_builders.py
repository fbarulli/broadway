from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.config.schema import DerivedFeature
from broadway.features.builders import _same_group, build_derived


def test_same_group_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        _same_group(df, "group", "group_lookup")


def test_build_derived_missing_source_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    feat = DerivedFeature(name="hour", func="datetime_hour", source="nope")
    with pytest.raises(ValueError):
        build_derived(df, [feat], "target")


def test_build_derived_unknown_func_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    feat = DerivedFeature(name="x", func="bogus", source="a")
    with pytest.raises(ValueError):
        build_derived(df, [feat], "target")


def test_build_derived_ok() -> None:
    df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 17:00"])})
    feat = DerivedFeature(name="h", func="datetime_hour", source="dt")
    result = build_derived(df, [feat], "target")
    assert "h" in result.columns
    assert result["h"].tolist() == [9, 17]


def test_build_derived_target_collision_raises() -> None:
    # GUARD: a derived feature may never overwrite the training target, even
    # when the target column is not physically present in the input frame.
    df = pd.DataFrame({"x": [1.0, 2.0]})
    feat = DerivedFeature(name="fare", func="log_distance", source="x")
    with pytest.raises(ValueError, match=r"collides with the dataset target 'fare'"):
        build_derived(df, [feat], "fare")


def test_build_derived_duplicate_names_raise() -> None:
    # GUARD: two derived entries sharing one name would make the later entry
    # silently overwrite the earlier one — must raise instead.
    df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01 09:00", "2024-01-06 17:00"])})
    feats = [
        DerivedFeature(name="h", func="datetime_hour", source="dt"),
        DerivedFeature(name="h", func="is_weekend", source="dt"),
    ]
    with pytest.raises(ValueError, match=r"previously built derived feature"):
        build_derived(df, feats, "t")


def test_build_derived_existing_column_collision_raises() -> None:
    # GUARD: a derived name colliding with a source data column would
    # clobber that column in place — must raise instead.
    df = pd.DataFrame({"x": [1.0, 2.0]})
    feat = DerivedFeature(name="x", func="log_distance", source="x")
    with pytest.raises(ValueError, match=r"existing column 'x' \(input data column\)"):
        build_derived(df, [feat], "t")


def test_build_derived_chaining_earlier_derived_source_allowed() -> None:
    # Chaining stays legitimate: consuming an EARLIER derived feature's
    # output as source must not trip the collision guard.
    df = pd.DataFrame({"x": [1.0, 2.0]})
    feats = [
        DerivedFeature(name="log_x", func="log_distance", source="x"),
        DerivedFeature(name="log_log_x", func="log_distance", source="log_x"),
    ]
    result = build_derived(df, feats, "t")
    expected = np.log1p(np.log1p(df["x"].to_numpy()))
    np.testing.assert_allclose(result["log_log_x"].to_numpy(), expected)
