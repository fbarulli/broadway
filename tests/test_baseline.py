from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broadway.stats.baseline import evaluate, train_lgbm


def _make_data(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = 2.0 * X["a"] - 1.0 * X["b"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_train_lgbm_returns_fitted_model() -> None:
    X, y = _make_data()
    model = train_lgbm(X, y, n_estimators=20, random_state=42, verbose=-1)

    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_evaluate_returns_mae_rmse_tail_mae() -> None:
    X, y = _make_data()
    model = train_lgbm(X, y, n_estimators=20, random_state=42, verbose=-1)

    result = evaluate(model, X, y, tail_quantile=0.9)

    assert set(result) == {"mae", "rmse", "tail_mae"}
    assert result["mae"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["tail_mae"] >= 0.0
    assert result["rmse"] >= result["mae"]


class _FixedPredictor:
    """Deterministic stub model returning a fixed prediction vector."""

    def __init__(self, preds: np.ndarray) -> None:
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._preds.copy()


def test_tail_mae_uses_top_quantile_only() -> None:
    # y = 1..10 with perfect predictions except the top slice: y=9 gets 6
    # (error -3) and y=10 gets 15 (error +5). np.quantile(y, 0.8) with the
    # default linear method lands at 8.2 (between 8 and 9), so the tail
    # slice is exactly {9, 10}.
    y = np.arange(1.0, 11.0)
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 15.0])
    model = _FixedPredictor(preds)
    X = pd.DataFrame({"x": np.arange(10.0)})

    result = evaluate(model, X, y, tail_quantile=0.8)

    # Hand-computed: tail MAE = (|-3| + |5|) / 2 = 4.0; all-rows MAE =
    # (0*8 + 3 + 5) / 10 = 0.8. abs=1e-9 sits far below the 3.2 gap to the
    # forbidden mean-over-all-rows value, so reverting to an unmasked mean
    # -- or shifting the slice by even one row -- lands RED, while staying
    # above float summation noise.
    assert result["tail_mae"] == pytest.approx(4.0, abs=1e-9)

    # Negative control: on identical inputs the unmasked number is
    # materially different, so the assertion above can only pass when the
    # top-quantile restriction is real.
    overall_mae = float(np.mean(np.abs(preds - y)))
    assert result["mae"] == pytest.approx(overall_mae, abs=1e-9)
    assert overall_mae == pytest.approx(0.8, abs=1e-9)
    assert abs(result["tail_mae"] - overall_mae) > 1.0
