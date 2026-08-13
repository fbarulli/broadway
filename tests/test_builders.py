from __future__ import annotations

import pandas as pd
import pytest

from broadway.config.schema import DerivedFeature
from broadway.features.builders import build_derived


def test_build_derived_missing_source_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    feat = DerivedFeature(name="h", func="datetime_hour", source="nope")
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
