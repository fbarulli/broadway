"""05: LightGBM baseline on every population — A/B/C comparison.

Trains the config LightGBM (native categorical features) on each population's
chronological split, evaluates on the future holdout (MAE / RMSE / R2 / tail
MAE via the platform metric helpers), and compares A (Manhattan-heavy) vs B
(borough-stratified) vs C (Manhattan + outer-borough weighting). Fails loudly
if a split parquet or a configured feature is missing (run 04 first).
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
    """Train + evaluate one population on its future holdout."""
    X_train, y_train, X_test, y_test, weights = load_split(cfg, spec)
    model = fit_lgbm(cfg, X_train, y_train, weights)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test.to_numpy(), preds)
    tail = evaluate(model, X_test, y_test.to_numpy(),
                    tail_quantile=TAIL_QUANTILE)
    result = {
        "population": name,
        "sample_name": spec["sample_name"],
        "source": spec["source"],
        "weighted": spec["weighting"]["enabled"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "tail_mae": tail["tail_mae"],
    }
    print(f"population {name}: MAE={result['mae']:.4f} "
          f"RMSE={result['rmse']:.4f} R2={result['r2']:.4f} "
          f"tail_MAE={result['tail_mae']:.4f}")
    return result


def plot_results(results: list[dict], out: Path) -> None:
    """Grouped MAE/RMSE bars across populations."""
    names = [r["population"] for r in results]
    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], [r["mae"] for r in results],
           width, label="MAE", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], [r["rmse"] for r in results],
           width, label="RMSE", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("$ (test holdout)")
    ax.set_title("LightGBM baselines — population A/B/C (total_amount)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)

    results = [evaluate_population(cfg, name, spec)
               for name, spec in cfg["baseline"]["populations"].items()]
    frame = pd.DataFrame(results)
    csv = RESULTS / f"{CSV_STEM}.csv"
    frame.to_csv(csv, index=False)
    print(f"wrote {csv}")

    out = RESULTS / "baseline_lightgbm.json"
    out.write_text(json.dumps({
        "method": ("LightGBM (native categoricals) per population, "
                   "chronological 80/20 holdout, target total_amount"),
        "populations": results,
    }, indent=2))
    print(f"wrote {out}")

    plot_results(results, RESULTS / f"{CSV_STEM}.png")
    print(f"wrote {CSV_STEM}.png")


if __name__ == "__main__":
    main()
