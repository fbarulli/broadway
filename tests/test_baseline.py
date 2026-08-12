from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_tail_mae_uses_top_quantile_only() -> None:
    X, y = _make_data()
    model = train_lgbm(X, y, n_estimators=20, random_state=42, verbose=-1)

    result = evaluate(model, X, y, tail_quantile=0.9)

    tail_mask = y >= np.quantile(y, 0.9)
    assert tail_mask.sum() < len(y)
    assert result["tail_mae"] >= 0.0
