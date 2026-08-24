"""Tests for robust-stats helpers and the coefficient-forest viz helper."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from broadway.stats.robust import (
    estimation_table,
    modified_zscore,
    outlier_mask,
    scenario_dollars,
    standardized_coefs,
    winsorize,
)
from broadway.viz import draw_coef_forest


def test_modified_zscore_matches_hand_computation() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    z = modified_zscore(series)

    # median = 3.5, MAD = 1.5 -> z(100) = 0.6745 * 96.5 / 1.5 ~ 43.393
    assert z.iloc[5] == pytest.approx(0.6745 * 96.5 / 1.5)
    assert z.iloc[0] == pytest.approx(0.6745 * -2.5 / 1.5)
    assert z.iloc[2] == pytest.approx(0.6745 * -0.5 / 1.5)


def test_modified_zscore_zeros_when_mad_is_zero() -> None:
    series = pd.Series([5.0, 5.0, 5.0, 5.0])

    z = modified_zscore(series)

    assert (z == 0.0).all()


def test_outlier_mask_flags_only_configured_column_extremes() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "b": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 100.0, 5.0],
        }
    )

    mask_a = outlier_mask(df, ["a"], threshold=3.5)
    mask_c = outlier_mask(df, ["c"], threshold=3.5)
    mask_both = outlier_mask(df, ["a", "c"], threshold=3.5)
    mask_b = outlier_mask(df, ["b"], threshold=3.5)

    assert mask_a.tolist() == [False] * 5 + [True]
    assert mask_c.tolist() == [False] * 4 + [True, False]
    assert mask_both.tolist() == [False] * 4 + [True, True]
    assert not mask_b.any()


def test_winsorize_caps_exactly_at_quantile_and_copies() -> None:
    df = pd.DataFrame({"a": list(range(1, 11)), "b": [5.5] * 10})

    out = winsorize(df, ["a"], cap_quantile=0.9)

    cap = df["a"].quantile(0.9)  # 9.1
    assert out["a"].max() == cap
    assert out["a"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.1])
    assert out["b"].tolist() == [5.5] * 10
    assert df["a"].max() == 10  # input frame untouched


def test_estimation_table_returns_four_columns() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.0 * x + rng.normal(scale=0.5, size=50)
    design = sm.add_constant(pd.DataFrame({"x": x}))
    model = sm.OLS(y, design).fit()

    table = estimation_table(model)
    # Independently computed HC3 reference, not the (nonrobust) fit's own bse.
    hc3 = model.get_robustcov_results("HC3")
    hc3_bse = pd.Series(hc3.bse, index=model.model.exog_names)

    assert list(table.columns) == ["coef", "HC3_SE", "CI_low", "CI_high"]
    assert set(table.index) == {"const", "x"}
    assert table.loc["x", "coef"] == pytest.approx(model.params["x"])
    assert table.loc["x", "HC3_SE"] == pytest.approx(hc3_bse["x"])
    assert (table["HC3_SE"] > 0).all()
    assert (table["CI_low"] < table["coef"]).all()
    assert (table["coef"] < table["CI_high"]).all()

    # Fail-loud: inputs without get_robustcov_results must not silently
    # fall back to plain (nonrobust) SEs labeled as HC3.
    class _NoRobustCovariance:
        pass

    with pytest.raises(TypeError, match="get_robustcov_results"):
        estimation_table(_NoRobustCovariance())


def test_estimation_table_plain_ols_still_yields_hc3_columns() -> None:
    # Landmine: a plain OLS fit (no cov_type) must NOT have its plain SEs
    # labeled HC3; estimation_table must produce true HC3 SEs and CIs
    # regardless of how the input was fitted.
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    y = 1.5 * x + rng.normal(scale=1.0, size=60)
    design = sm.add_constant(pd.DataFrame({"x": x}))
    plain = sm.OLS(y, design).fit()  # no cov_type: nonrobust fit

    table = estimation_table(plain)
    hc3 = plain.get_robustcov_results("HC3")
    hc3_ci = pd.DataFrame(hc3.conf_int(alpha=0.05), index=plain.model.exog_names)

    # HC3 SEs differ from the plain fit's own bse...
    assert not np.allclose(
        table["HC3_SE"].to_numpy(), plain.bse.to_numpy(), rtol=0.0, atol=0.0
    )
    # ...and match an independently computed HC3 fit exactly.
    assert table["HC3_SE"].to_numpy() == pytest.approx(hc3.bse)
    assert table["CI_low"].to_numpy() == pytest.approx(
        hc3_ci.iloc[:, 0].to_numpy()
    )
    assert table["CI_high"].to_numpy() == pytest.approx(
        hc3_ci.iloc[:, 1].to_numpy()
    )


def test_standardized_coefs_matches_hand_computed_values() -> None:
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    y = 2.0 * x1 + 0.5 * x2
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    model = sm.OLS(y, sm.add_constant(df[["x1", "x2"]])).fit()

    result = standardized_coefs(model, df, ["x1", "x2"], "y")

    sd_x1 = float(np.std(x1, ddof=1))
    sd_x2 = float(np.std(x2, ddof=1))
    sd_y = float(np.std(y, ddof=1))

    assert result["x1"]["coef"] == pytest.approx(2.0)
    assert result["x2"]["coef"] == pytest.approx(0.5)
    assert result["x1"]["sd_x"] == pytest.approx(sd_x1)
    assert result["x2"]["sd_x"] == pytest.approx(sd_x2)
    assert result["x1"]["sd_y"] == pytest.approx(sd_y)
    assert result["x1"]["beta_std"] == pytest.approx(2.0 * sd_x1 / sd_y)
    assert result["x2"]["beta_std"] == pytest.approx(0.5 * sd_x2 / sd_y)


def test_scenario_dollars_multiplies_change_by_coef() -> None:
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    y = 2.0 * x1 + 0.5 * x2
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    model = sm.OLS(y, sm.add_constant(df[["x1", "x2"]])).fit()

    rows = scenario_dollars(
        model, [("x2 +2", "x2", 2.0), ("x1 +1", "x1", 1.0)]
    )

    assert rows[0] == {
        "label": "x2 +2",
        "term": "x2",
        "change": 2.0,
        "dollars": pytest.approx(1.0),  # 2.0 * 0.5
    }
    assert rows[1]["dollars"] == pytest.approx(2.0)  # 1.0 * 2.0
    assert set(rows[1]) == {"label", "term", "change", "dollars"}


def test_draw_coef_forest_structure_and_annotations() -> None:
    coef = pd.Series([1.5, -0.5], index=["x1", "x2"])
    ci_low = pd.Series([0.8, -1.2], index=["x1", "x2"])
    ci_high = pd.Series([2.2, 0.2], index=["x1", "x2"])

    fig, ax = plt.subplots()
    draw_coef_forest(ax, coef, ci_low, ci_high, labels={"x1": "X One"})
    plt.close(fig)

    assert np.allclose(ax.lines[0].get_xdata(), [1.5, -0.5])
    assert len(ax.lines) >= 2  # marker line + zero reference line
    assert len(ax.collections) >= 1  # CI error-bar segments
    assert any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in ax.lines)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["X One", "x2"]
    assert len(ax.texts) == 2
    assert ax.texts[0].get_text() == "1.500 [0.80, 2.20]"


def test_draw_coef_forest_default_labels_and_no_annotations() -> None:
    coef = pd.Series([1.5, -0.5], index=["x1", "x2"])
    ci_low = pd.Series([0.8, -1.2], index=["x1", "x2"])
    ci_high = pd.Series([2.2, 0.2], index=["x1", "x2"])

    fig, ax = plt.subplots()
    draw_coef_forest(ax, coef, ci_low, ci_high, annotate=False)
    plt.close(fig)

    assert [t.get_text() for t in ax.get_yticklabels()] == ["x1", "x2"]
    assert len(ax.texts) == 0
