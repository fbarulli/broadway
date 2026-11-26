from __future__ import annotations

import numpy as np

from broadway.stats.changepoint import ChangePointMethod, cusum_changepoints


def test_cusum_finds_a_synthetic_mean_shift() -> None:
    rng = np.random.default_rng(7)
    series = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(5.0, 1.0, 300)])

    result = cusum_changepoints(series, threshold=5.0, min_size=24)

    assert result.method is ChangePointMethod.CUSUM
    assert len(result.change_indices) == 1
    assert abs(result.change_indices[0] - 300) <= 24
