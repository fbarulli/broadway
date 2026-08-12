from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.stats.base import stratified_sample


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_group = 100
    return pd.DataFrame(
        {
            "group": np.repeat(["a", "b", "c", "d"], n_per_group),
            "value": rng.normal(0.0, 1.0, n_per_group * 4),
        }
    )


def test_stratified_sample_preserves_row_count() -> None:
    df = _df()
    frac = 0.3
    out = stratified_sample(df, group_col="group", frac=frac, random_state=42)
    assert len(out) == pytest.approx(frac * len(df), rel=0.15)
    assert out.index.is_monotonic_increasing


def test_stratified_sample_keeps_per_group_proportions() -> None:
    df = _df()
    out = stratified_sample(df, group_col="group", frac=0.25, random_state=7)
    counts = out["group"].value_counts(normalize=True).sort_index()
    expected = df["group"].value_counts(normalize=True).sort_index()
    assert np.allclose(counts.values, expected.values, atol=0.05)


def test_stratified_sample_is_deterministic() -> None:
    df = _df()
    a = stratified_sample(df, group_col="group", frac=0.2, random_state=1)
    b = stratified_sample(df, group_col="group", frac=0.2, random_state=1)
    pd.testing.assert_frame_equal(a, b)
