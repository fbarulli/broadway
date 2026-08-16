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

from _setup import (
    RESULTS,
    WORKING_DATASET,
    build_borough_dummies,
    load_config,
    load_manhattan_sample,
    require_finite,
)
from broadway.stats.regression import bp_jb

CSV_STEM = Path(__file__).stem
METRICS = ("kurtosis", "skew", "jb_stat", "bp_stat", "rsquared")


def fit_with_dummies(df: pd.DataFrame, cfg: dict, use_borough: bool):
    """OLS on the target; optional borough one-hot dummies (config reference).

    Fails loudly on NaN/Inf or row-count mismatch instead of fitting silently
    on misaligned input (the two bugs this experiment actually hit).
    """
    spec = cfg["borough_dummies"]
    if use_borough:
        col = spec["column"]
        if spec["reference"] not in set(df[col].dropna().unique()):
            raise ValueError(f"reference borough '{spec['reference']}' "
                             f"not present in '{col}'")
        df = df[df[col].notna()]  # drop rows with an unmapped zone (no dummy)
    X = df[spec["features"]].copy()
    if use_borough:
        X = pd.concat([X, build_borough_dummies(df, cfg)], axis=1)
    X = sm.add_constant(X)
    if len(X) != len(df):
        raise ValueError(f"exog/endog row mismatch ({len(X)} vs {len(df)})")
    require_finite(X, f"fit use_borough={use_borough}")
    return sm.OLS(df[cfg["target"]], X).fit()


def summarize(model) -> dict:
    """Residual diagnostics (kurtosis/skew/JB/BP) + R^2 for one fit."""
    diag = bp_jb(model)
    return {
        "rsquared": float(model.rsquared),
        "kurtosis": diag["kurtosis"],
        "skew": diag["skew"],
        "jb_stat": diag["jb_stat"],
        "bp_stat": diag["bp_stat"],
        "n_params": int(model.params.size),
    }


def main() -> None:
    cfg = load_config()
    df = load_manhattan_sample(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)

    before = summarize(fit_with_dummies(df, cfg, use_borough=False))
    after = summarize(fit_with_dummies(df, cfg, use_borough=True))

    spec = cfg["borough_dummies"]
    print(f"sample: {cfg['sample']['name']} (pickup = "
          f"{cfg['sample']['pickup_borough']}, n = {len(df)})")
    print(f"{spec['column']} dummies: reference = {spec['reference']}")
    print(f"\n{'metric':<10}{'without':>12}{'with':>12}{'delta':>12}")
    for m in METRICS:
        print(f"{m:<10}{before[m]:>12.4g}{after[m]:>12.4g}"
              f"{after[m] - before[m]:>+12.4g}")

    deltas = {m: round(float(after[m] - before[m]), 4) for m in METRICS}
    payload = {
        "method": (f"raw fare ~ distance + duration with/without "
                   f"{spec['column']} one-hot dummies (plain OLS); "
                   f"residual heavy-tail test on {cfg['sample']['name']}"),
        "n": int(len(df)),
        "reference_borough": spec["reference"],
        "without_borough": before,
        "with_borough": after,
        "deltas": deltas,
    }

    out = RESULTS / f"{cfg['sample']['name']}.json"
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
