"""05: LightGBM baseline on every population — A/B/C comparison.

Trains the config LightGBM (native categorical features) on each population's
chronological split, evaluates on the future holdout (MAE / RMSE / R2 / tail
MAE via the platform metric helpers), and renders a performance dashboard:
metric bars incl. R2, overlaid residual histograms, actual-vs-predicted and
residual-vs-fitted panels per population. Fails loudly if a split parquet or
a configured feature is missing (run 04 first).
"""

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _setup import RESULTS, load_config

from broadway.evaluate.metrics import compute_metrics
from broadway.stats.baseline import evaluate

CSV_STEM = Path(__file__).stem
TAIL_QUANTILE = 0.95
COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def load_split(cfg: dict, spec: dict) -> tuple:
    """Read one population's split; fill missing categoricals; return X/y + weights."""
    sample_name = spec["sample_name"]
    train = pd.read_parquet(RESULTS / f"{sample_name}_train.parquet")
    test = pd.read_parquet(RESULTS / f"{sample_name}_test.parquet")
    features = cfg["baseline"]["features"]
    target = cfg["baseline"]["target"]
    missing = [f for f in features
               if f not in train.columns or f not in test.columns]
    if missing:
        raise ValueError(f"population {sample_name}: missing feature(s) in "
                         f"split parquet: {missing}")
    for frame in (train, test):
        for col in cfg["baseline"]["categorical_features"]:
            frame[col] = (frame[col].fillna(cfg["baseline"]["missing_label"])
                          .astype("category"))  # LightGBM native categoricals

    weights = None
    if spec["weighting"]["enabled"]:
        if "sample_weight" not in train.columns:
            raise ValueError(f"population {sample_name}: weighting enabled but "
                             "no sample_weight column in split")
        weights = train["sample_weight"].to_numpy()
    return (train[features], train[target], test[features], test[target],
            weights)


def fit_lgbm(cfg: dict, X_train, y_train, weights) -> object:
    """Config LightGBM with native categorical features."""
    params = dict(cfg["baseline"]["lgbm"])
    params["random_state"] = cfg["baseline"]["seed"]
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight=weights,
              categorical_feature=cfg["baseline"]["categorical_features"])
    return model


def evaluate_population(cfg: dict, name: str, spec: dict) -> dict:
    """Train + evaluate one population; returns result + predictions."""
    X_train, y_train, X_test, y_test, weights = load_split(cfg, spec)
    model = fit_lgbm(cfg, X_train, y_train, weights)
    preds = model.predict(X_test)
    actuals = y_test.to_numpy()
    metrics = compute_metrics(actuals, preds)
    tail = evaluate(model, X_test, actuals, tail_quantile=TAIL_QUANTILE)
    result = {
        "population": name,
        "sample_name": spec["sample_name"],
        "source": spec["source"],
        "weighted": spec["weighting"]["enabled"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "tail_mae": tail["tail_mae"],
    }
    print(f"population {name}: MAE={result['mae']:.4f} "
          f"RMSE={result['rmse']:.4f} R2={result['r2']:.4f} "
          f"tail_MAE={result['tail_mae']:.4f}")
    return {"result": result, "actuals": actuals, "preds": preds}


def _bar(ax, names: list[str], values: list[float], colors: list[str],
         ylabel: str, title: str, decimals: int = 4) -> None:
    x = range(len(names))
    ax.bar(list(x), values, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.{decimals}f}", xy=(i, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def plot_dashboard(results: list[dict], out: Path) -> None:
    """2x2 performance dashboard: MAE/RMSE, R2, tail MAE, residual histograms."""
    names = [r["result"]["population"] for r in results]
    colors = [COLORS[n] for n in names]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
    ax1.bar([i - 0.18 for i in range(len(names))],
            [r["result"]["mae"] for r in results], 0.36, label="MAE",
            color=colors)
    ax1.bar([i + 0.18 for i in range(len(names))],
            [r["result"]["rmse"] for r in results], 0.36, label="RMSE",
            color="#7f7f7f")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("$ (holdout)")
    ax1.set_title("MAE / RMSE")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    _bar(ax2, names, [r["result"]["r2"] for r in results], colors,
         "R2", "R2 (holdout)")
    _bar(ax3, names, [r["result"]["tail_mae"] for r in results], colors,
         "$", "Tail MAE (top 5% of fares)")
    for r in results:
        resid = r["actuals"] - r["preds"]
        ax4.hist(resid, bins=60, alpha=0.45,
                 color=COLORS[r["result"]["population"]],
                 label=r["result"]["population"])
    ax4.set_xlabel("residual ($)")
    ax4.set_ylabel("count")
    ax4.set_title("Residual distribution (holdout)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")
    fig.suptitle("LightGBM baselines — performance by population "
                 "(total_amount, future holdout)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(results: list[dict], out: Path) -> None:
    """Actual vs predicted scatter per population with the y=x reference."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, r in zip(axes, results):
        pop = r["result"]["population"]
        ax.scatter(r["preds"], r["actuals"], s=4, alpha=0.25,
                   color=COLORS[pop])
        lim = [min(r["actuals"].min(), r["preds"].min()),
               max(r["actuals"].max(), r["preds"].max())]
        ax.plot(lim, lim, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("predicted ($)")
        ax.set_ylabel("actual ($)")
        ax.set_title(f"population {pop} — MAE ${r['result']['mae']:.2f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Actual vs predicted (holdout)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_residuals(results: list[dict], out: Path) -> None:
    """Residuals vs predicted per population (the fan-out view)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, r in zip(axes, results):
        pop = r["result"]["population"]
        resid = r["actuals"] - r["preds"]
        ax.scatter(r["preds"], resid, s=4, alpha=0.25, color=COLORS[pop])
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("predicted ($)")
        ax.set_ylabel("residual ($)")
        ax.set_title(f"population {pop} — residuals vs fitted")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Residuals vs predicted (holdout)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    items = [evaluate_population(cfg, name, spec)
             for name, spec in cfg["baseline"]["populations"].items()]
    frame = pd.DataFrame([i["result"] for i in items])
    csv = RESULTS / f"{CSV_STEM}.csv"
    frame.to_csv(csv, index=False)
    print(f"wrote {csv}")

    out = RESULTS / "baseline_lightgbm.json"
    out.write_text(json.dumps({
        "method": ("LightGBM (native categoricals) per population, "
                   "chronological 80/20 holdout, target total_amount"),
        "populations": [i["result"] for i in items],
    }, indent=2))
    print(f"wrote {out}")

    plot_dashboard(items, RESULTS / f"{CSV_STEM}_performance.png")
    plot_actual_vs_predicted(items, RESULTS / f"{CSV_STEM}_actual_vs_pred.png")
    plot_residuals(items, RESULTS / f"{CSV_STEM}_residuals.png")
    print("wrote performance / actual_vs_pred / residuals figures")


if __name__ == "__main__":
    main()
