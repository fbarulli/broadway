"""02: does geography eat the heavy tails? — pickup_borough dummies vs kurtosis.

Fits raw fare ~ distance + duration with and without pickup_borough one-hot
dummies (reference borough from config) and compares residual kurtosis /
Jarque-Bera / Breusch-Pagan / R^2 — the direct test of whether geography
explains the heavy tails chased since steps 26-28. Merges a borough_dummies
block into the multivariate dataset JSON and writes a tracked CSV.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from _setup import RESULTS, WORKING_DATASET, load_config, load_metered_categorical
from broadway.stats.regression import bp_jb

CSV_STEM = Path(__file__).stem
METRICS = ("kurtosis", "jb_stat", "bp_stat", "rsquared")


def fit_with_dummies(df: pd.DataFrame, cfg: dict, use_borough: bool):
    """OLS on the target; optional borough one-hot dummies (config reference)."""
    spec = cfg["borough_dummies"]
    X = df[spec["features"]].copy()
    if use_borough:
        col = cfg["borough"]["column"]
        categories = [spec["reference"]] + sorted(
            v for v in df[col].dropna().unique() if v != spec["reference"])
        dummies = pd.get_dummies(pd.Categorical(df[col], categories=categories),
                                 prefix="borough", drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X = sm.add_constant(X)
    return sm.OLS(df[cfg["target"]], X).fit()


def summarize(model) -> dict:
    """Residual diagnostics (kurtosis/JB/BP) + R^2 for one fit."""
    diag = bp_jb(model)
    return {
        "rsquared": float(model.rsquared),
        "kurtosis": diag["kurtosis"],
        "jb_stat": diag["jb_stat"],
        "bp_stat": diag["bp_stat"],
        "n_params": int(model.params.size),
    }


def main() -> None:
    cfg = load_config()
    df = load_metered_categorical(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)

    before = summarize(fit_with_dummies(df, cfg, use_borough=False))
    after = summarize(fit_with_dummies(df, cfg, use_borough=True))

    print(f"borough dummies: reference = {cfg['borough_dummies']['reference']} "
          f"| n = {len(df)}")
    print(f"\n{'metric':<10}{'without':>12}{'with':>12}{'delta':>12}")
    for m in METRICS:
        print(f"{m:<10}{before[m]:>12.4g}{after[m]:>12.4g}"
              f"{after[m] - before[m]:>+12.4g}")

    deltas = {m: round(float(after[m] - before[m]), 4) for m in METRICS}
    payload = {
        "method": ("raw fare ~ distance + duration with/without pickup_borough "
                   "one-hot dummies (plain OLS); residual heavy-tail test"),
        "n": int(len(df)),
        "reference_borough": cfg["borough_dummies"]["reference"],
        "without_borough": before,
        "with_borough": after,
        "deltas": deltas,
    }

    out = RESULTS / f"{WORKING_DATASET.stem}.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data["dataset"] = WORKING_DATASET.name
    data["source_script"] = Path(__file__).name
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["target"] = cfg["target"]
    data["borough_dummies"] = payload
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")

    csv = RESULTS / f"{CSV_STEM}.csv"
    pd.DataFrame([
        {"metric": m, "without_borough": before[m],
         "with_borough": after[m], "delta": deltas[m]}
        for m in METRICS
    ]).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
