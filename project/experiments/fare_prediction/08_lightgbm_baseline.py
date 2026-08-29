"""08: LightGBM baseline on strictly pre-trip features, log-dollar target, dollar-space metrics."""

from math import sqrt
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _common import RESULTS, SAFE_CATEGORICAL_FEATURES, SAFE_FEATURES, SEED
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREPARED_DIR = RESULTS / "prepared"
TARGET = "fare_amount"
TARGET_LOG = "fare_amount_log"
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    # Config-carried (project/config/experiments/fare_prediction.yaml) — ledger item f.
    "random_state": SEED,
    "verbosity": -1,
}
METRICS_CSV = RESULTS / "08_lightgbm_baseline_describe.csv"
PNG_IMPORTANCE = RESULTS / "08_lightgbm_baseline_importance.png"
PNG_FIT = RESULTS / "08_lightgbm_baseline_fit.png"
SUMMARY_MD = RESULTS / "08_lightgbm_baseline.md"
MODEL_PATH = RESULTS / "models" / "lgbm_baseline.txt"
FIT_SAMPLE = 25_000
TOP_FEATURES = 10


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


def plot_importance(importance: pd.Series, out_path: Path) -> None:
    """Top-N gain-importance barplot (share of total gain, %)."""
    top = importance.head(TOP_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#4c72b0")
    ax.set_xlabel("gain importance (% of total)")
    ax.set_ylabel("")
    ax.set_title("LightGBM feature importance (gain) — top-10")
    ax.grid(True, alpha=0.3, axis="x")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit(test: pd.DataFrame, preds_log: np.ndarray, out_path: Path) -> None:
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render an indexed frame as a Markdown pipe table with ``:,.2f`` cells."""
    lines = [
        "| " + " | ".join(["split", *frame.columns]) + " |",
        "| " + " | ".join(["---"] * (len(frame.columns) + 1)) + " |",
    ]
    lines += [
        "| " + " | ".join([str(idx), *[f"{value:,.2f}" for value in row]]) + " |"
        for idx, row in frame.iterrows()
    ]
    return "\n".join(lines)


def summary_md(metrics: pd.DataFrame, importance: pd.Series) -> str:
    """Render the 08 Markdown summary: dollar-space metrics + top-10 gain features."""
    top = importance.head(TOP_FEATURES)
    rows = "\n".join(f"| {name} | {pct:.1f}% |" for name, pct in top.items())
    return "\n".join([
        "# LightGBM baseline (pre-trip features, log-dollar target)",
        "",
        (
            "LightGBM regressor on strictly pre-trip features; target "
            "`fare_amount_log`; MAE/RMSE/R² computed on real dollars via `np.expm1`."
        ),
        "",
        "## Dollar-space metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Top-10 gain importance",
        "",
        "| feature | gain (% of total) |",
        "| --- | --- |",
        rows,
        "",
    ])


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

    rows = []
    for name, frame in (("val", val), ("test", test)):
        metrics = evaluate(frame[TARGET], model.predict(frame[SAFE_FEATURES]))
        rows.append({"split": name, **metrics})
        print(f"{name}: MAE=${metrics['mae']:,.2f} "
              f"RMSE=${metrics['rmse']:,.2f} R2={metrics['r2']:.4f}")
    metrics_df = pd.DataFrame(rows).set_index("split").round(2)
    metrics_df.to_csv(METRICS_CSV)
    print(f"wrote {METRICS_CSV}")

    importance = gain_importance(model)
    plot_importance(importance, PNG_IMPORTANCE)
    print(f"wrote {PNG_IMPORTANCE}")
    print("top-10 gain features (%):")
    print(importance.head(TOP_FEATURES).round(2).to_string())

    plot_fit(test, model.predict(test[SAFE_FEATURES]), PNG_FIT)
    print(f"wrote {PNG_FIT}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(MODEL_PATH)
    print(f"wrote {MODEL_PATH}")

    SUMMARY_MD.write_text(summary_md(metrics_df, importance))
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
