"""Tests for broadway.evaluate.explain (generic model-explainability helpers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from broadway.evaluate import explain


def _synthetic_data() -> tuple[pd.DataFrame, np.ndarray, LinearRegression]:
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        {
            "alpha": rng.normal(0.0, 1.0, n),
            "beta": rng.normal(1.0, 2.0, n),
            "gamma": rng.uniform(-1.0, 1.0, n),
        }
    )
    y = 2.0 * X["alpha"] - X["beta"] + 0.5 * X["gamma"] + rng.normal(0.0, 0.1, n)
    model = LinearRegression().fit(X, y)
    return X, y.to_numpy(), model


def _assert_nonempty(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_shap_summary_kernel_writes_figure(tmp_path) -> None:
    X, _, model = _synthetic_data()
    out = tmp_path / "shap_kernel.png"
    explain.shap_summary(model, X, out, kind="kernel")
    _assert_nonempty(out)


def test_shap_summary_tree_writes_figure(tmp_path) -> None:
    X, y, _ = _synthetic_data()
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    out = tmp_path / "shap_tree.png"
    explain.shap_summary(tree, X, out, kind="tree")
    _assert_nonempty(out)


def test_shap_summary_invalid_kind_raises(tmp_path) -> None:
    X, _, model = _synthetic_data()
    with pytest.raises(ValueError):
        explain.shap_summary(model, X, tmp_path / "shap_bad.png", kind="bogus")


def test_permutation_importance_table_shape() -> None:
    X, y, model = _synthetic_data()
    table = explain.permutation_importance_table(
        model, X, y, n_repeats=5, random_state=0
    )
    assert list(table.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(table) == X.shape[1]
    assert list(table["feature"]) == list(X.columns)


def test_pdp_ice_writes_figure(tmp_path) -> None:
    X, _, model = _synthetic_data()
    out = tmp_path / "pdp_ice.png"
    explain.pdp_ice(model, X, ["alpha", "beta"], out)
    _assert_nonempty(out)


def test_lime_explanation_writes_figure(tmp_path) -> None:
    X, _, model = _synthetic_data()
    out = tmp_path / "lime.png"
    explain.lime_explanation(model, X, X.iloc[0], list(X.columns), out)
    _assert_nonempty(out)


def test_residual_plot_writes_figure(tmp_path) -> None:
    X, y, model = _synthetic_data()
    out = tmp_path / "residuals.png"
    explain.residual_plot(model.predict(X), y, out)
    _assert_nonempty(out)
