from __future__ import annotations

import pytest

from broadway.selection.information import compare_aic_bic


class _Fit:
    """Minimal duck type of a fitted statsmodels result (aic/bic only)."""

    def __init__(self, aic: float, bic: float) -> None:
        self.aic = aic
        self.bic = bic


def test_compare_aic_bic_ranks_by_aic_with_deltas_and_weights() -> None:
    table = compare_aic_bic(
        {
            "wide": _Fit(aic=130.0, bic=150.0),
            "narrow": _Fit(aic=100.0, bic=112.0),
            "medium": _Fit(aic=110.0, bic=130.0),
        }
    )

    assert [candidate.name for candidate in table.candidates] == ["narrow", "medium", "wide"]
    assert table.best_by_aic == "narrow"
    assert table.best_by_bic == "narrow"
    assert table.candidates[0].delta_aic == pytest.approx(0.0)
    assert table.candidates[0].delta_bic == pytest.approx(0.0)
    assert table.candidates[1].delta_aic == pytest.approx(10.0)
    assert table.candidates[1].delta_bic == pytest.approx(18.0)
    assert sum(candidate.akaike_weight for candidate in table.candidates) == pytest.approx(1.0)
    assert sum(candidate.bic_weight for candidate in table.candidates) == pytest.approx(1.0)
