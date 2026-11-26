"""08: LightGBM baseline on strictly pre-trip features, log-dollar target, dollar-space metrics.

Answers whether the baseline model beats the naive mean: relative RMSE
improvement on the untouched test split.
"""

from math import sqrt
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _common import (
    RESULTS,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_FEATURES,
    SEED,
    load_decision_rule,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput

PREPARED_DIR = RESULTS / "prepared"
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
STEM = "08_lightgbm_baseline"
ANALYSIS_SAMPLE = 50_000
BLOCK = "day"
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    # Config-carried (project/config/experiments/fare_prediction.yaml) — ledger item f.
    "random_state": SEED,
    "verbosity": -1,
}
PNG_IMPORTANCE = RESULTS / "08_lightgbm_baseline_importance.png"
PNG_FIT = RESULTS / "08_lightgbm_baseline_fit.png"
MODEL_PATH = RESULTS / "models" / "lgbm_baseline.txt"
FIT_SAMPLE = 25_000
TOP_FEATURES = 10
BOOTSTRAP_ITERATIONS = 200


def evaluate(y_dollars: pd.Series, preds_log: np.ndarray) -> dict[str, float]:
    """Dollar-space metrics from log-dollar predictions (``np.expm1`` back-transform)."""
    preds_dollars = np.expm1(preds_log)
    return {
        "mae": mean_absolute_error(y_dollars, preds_dollars),
        "rmse": sqrt(mean_squared_error(y_dollars, preds_dollars)),
        "r2": r2_score(y_dollars, preds_dollars),
    }


def _as_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the raw location-id columns to LightGBM native categoricals."""
    out = frame.copy()
    for col in SAFE_CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    return out


def gain_importance(model: lgb.LGBMRegressor) -> pd.Series:
    """Gain-importance series as % of total gain, features sorted descending."""
    gain = model.booster_.feature_importance("gain")
    total = float(gain.sum())
    importance = pd.Series(gain / total * 100.0, index=model.booster_.feature_name())
    return importance.sort_values(ascending=False)


def plot_importance(importance: pd.Series) -> object:
    """Top-N gain-importance barplot (share of total gain, %)."""
    top = importance.head(TOP_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4c72b0")
    ax.set_xlabel("gain importance (% of total)")
    ax.set_ylabel("")
    ax.set_title("LightGBM feature importance (gain) — top-10")
    ax.grid(True, alpha=0.3, axis="x")
    return fig


def plot_fit(test: pd.DataFrame, preds_log: np.ndarray) -> object:
    """Predicted vs actual dollars on TEST (downsampled), with y=x reference line."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), min(FIT_SAMPLE, len(test)), replace=False)
    preds = np.expm1(preds_log)[idx]
    actuals = test[TARGET].to_numpy()[idx]
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(preds, actuals, s=4, alpha=0.25, color="#4c72b0")
    lim = [min(preds.min(), actuals.min()), max(preds.max(), actuals.max())]
    ax.plot(lim, lim, "--", color="#d62728", lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted fare ($)")
    ax.set_ylabel("actual fare ($)")
    ax.set_title("Predicted vs actual fare (test) — LightGBM baseline")
    ax.grid(True, alpha=0.3)
    return fig


def plot_model_vs_naive(metrics: pd.DataFrame) -> object:
    """Bars of test RMSE: naive train-mean baseline vs the LightGBM model."""
    test = metrics.loc["test"]
    values = {"naive mean": float(test["baseline_rmse"]), "LightGBM": float(test["rmse"])}
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bars = ax.bar(list(values), list(values.values()), color=["#9ecae1", "#08519c"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${bar.get_height():,.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("test RMSE ($)")
    ax.set_title("model vs naive mean — lower is better")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def _improvement(frame: pd.DataFrame) -> float:
    """Relative RMSE improvement of ``predicted`` over the naive ``baseline``."""
    actual = frame["actual"].to_numpy()
    model_rmse = sqrt(mean_squared_error(actual, frame["predicted"].to_numpy()))
    naive_rmse = sqrt(mean_squared_error(actual, frame["baseline"].to_numpy()))
    return 1.0 - model_rmse / naive_rmse


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = _as_categorical(pd.read_parquet(PREPARED_DIR / "train.parquet"))
    val = _as_categorical(pd.read_parquet(PREPARED_DIR / "val.parquet"))
    test = _as_categorical(pd.read_parquet(PREPARED_DIR / "test.parquet"))

    model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(
        train[SAFE_FEATURES], train[TARGET_LOG],
        eval_set=[(val[SAFE_FEATURES], val[TARGET_LOG])],
        categorical_feature=SAFE_CATEGORICAL_FEATURES,
    )

    naive_mean = float(train[TARGET].mean())
    rows = []
    test_pred = None
    for name, frame in (("val", val), ("test", test)):
        preds_log = model.predict(frame[SAFE_FEATURES])
        metrics = evaluate(frame[TARGET], preds_log)
        baseline_rmse = sqrt(mean_squared_error(frame[TARGET], np.full(len(frame), naive_mean)))
        rows.append({
            "split": name,
            **metrics,
            "baseline_rmse": baseline_rmse,
            "improvement": 1.0 - metrics["rmse"] / baseline_rmse,
        })
        print(f"{name}: MAE=${metrics['mae']:,.2f} RMSE=${metrics['rmse']:,.2f} "
              f"R2={metrics['r2']:.4f} (naive RMSE ${baseline_rmse:,.2f})")
        if name == "test":
            test_pred = np.expm1(preds_log)
    metrics_df = pd.DataFrame(rows).set_index("split").round(4)

    importance = gain_importance(model)
    print("top-10 gain features (%):")
    print(importance.head(TOP_FEATURES).round(2).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(MODEL_PATH)
    print(f"wrote {MODEL_PATH}")

    actual = test[TARGET].to_numpy()
    predicted = np.asarray(test_pred, dtype=float)
    baseline = np.full(len(test), naive_mean)
    frame = test.assign(
        actual=actual,
        predicted=predicted,
        baseline=baseline,
        day=test["pickup_datetime"].dt.floor("D"),
    )
    analysis = frame.sample(n=min(ANALYSIS_SAMPLE, len(frame)), random_state=SEED)
    weekend = (analysis["pickup_day_of_week"] >= 5).to_numpy()
    half = frame.sample(n=max(2, len(analysis) // 2), random_state=SEED + 1)
    rule = load_decision_rule(STEM)
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="actual",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        analysis, _improvement, spec, name=rule.metric, unit="relative_improvement"
    )
    n = measurement.sample_size
    decision_evidence = Evidence(
        baseline="naive train-mean prediction (the model must beat it out of sample)",
        segments=[
            SegmentEvidence(
                segment="weekday",
                value=_improvement(analysis[~weekend]),
                sample_size=int((~weekend).sum()),
            ),
            SegmentEvidence(
                segment="weekend",
                value=_improvement(analysis[weekend]),
                sample_size=int(weekend.sum()),
            ),
        ],
        sensitivity={
            f"mae_improvement_n{n}": 1.0 - float(metrics_df.loc["test", "mae"]) / float(
                np.mean(np.abs(actual - naive_mean))
            ),
            f"val_improvement_n{len(val)}": float(metrics_df.loc["val", "improvement"]),
            f"improvement_second_seed_n{len(half)}": _improvement(half),
        },
        falsifier="the RMSE-improvement interval falls below the 5% floor on test",
        provenance={
            "artifact": MODEL_PATH.name,
            "source": str(PREPARED_DIR.name),
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "post-trip columns (distance/duration/speed) excluded as leakage",
        },
    )
    decision = measurement.to_decision(
        rule,
        evidence=decision_evidence,
        question="Does the baseline model beat the naive mean out of sample?",
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("describe", metrics_df.reset_index())
    output.add_table("importance", importance.head(TOP_FEATURES).reset_index())
    output.save_figure(plot_model_vs_naive(metrics_df))
    plt.close("all")
    plot_importance(importance).savefig(PNG_IMPORTANCE, dpi=150)
    plot_fit(test, model.predict(test[SAFE_FEATURES])).savefig(PNG_FIT, dpi=150)
    plt.close("all")
    report, figure, decision_path = output.write()
    print(f"report: {report}")
    print(f"plot: {figure}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}], "
        f"n={measurement.sample_size:,}, null={decision.null_threshold})"
    )


if __name__ == "__main__":
    main()

