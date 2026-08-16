"""03: the geography premium test — dropoff_borough HC3 premiums vs Manhattan.

On the manhattan_sample, refits the baseline (fare ~ distance + duration)
with dropoff_borough one-hot dummies (Manhattan reference) and reports the
dollar premium per borough with HC3 95% CIs, plus residual kurtosis / JB /
BP / R^2 before and after — does destination geography carry a fare premium
after controlling distance and duration? Missing boroughs are filled with a
config label (distinct from the real 'Unknown' zone value). Note: the
reference is EXPLICIT (config), not statsmodels' alphabetical default.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from _setup import (
    RESULTS,
    WORKING_DATASET,
    build_borough_dummies,
    load_config,
    load_manhattan_sample,
)
from broadway.stats.regression import bp_jb
from broadway.utils import require_finite

CSV_STEM = Path(__file__).stem
BOROUGH_PREFIX = "borough_"
DIAGNOSTIC_METRICS = ("kurtosis", "skew", "jb_stat", "bp_stat", "rsquared")


def diagnostics(model) -> dict:
    """Residual diagnostics (kurtosis/skew/JB/BP) + R^2 for one HC3 fit."""
    diag = bp_jb(model)
    return {
        "rsquared": float(model.rsquared),
        "kurtosis": diag["kurtosis"],
        "skew": diag["skew"],
        "jb_stat": diag["jb_stat"],
        "bp_stat": diag["bp_stat"],
    }


def fit_baseline(df: pd.DataFrame, cfg: dict):
    """HC3 OLS of target on the config features (no borough)."""
    X = sm.add_constant(df[cfg["borough_dummies"]["features"]])
    require_finite(X, "baseline fit")
    return sm.OLS(df[cfg["target"]], X).fit(cov_type="HC3")


def fit_geo(df: pd.DataFrame, cfg: dict):
    """HC3 OLS of target on features + borough dummies (config reference)."""
    spec = cfg["borough_dummies"]
    X = pd.concat([df[spec["features"]], build_borough_dummies(df, cfg)], axis=1)
    X = sm.add_constant(X)
    require_finite(X, "geography premium fit")
    return sm.OLS(df[cfg["target"]], X).fit(cov_type="HC3")


def premium_table(model, cfg: dict) -> pd.DataFrame:
    """Per-borough dollar premium vs the reference (reference = $0.00)."""
    rows = []
    for term in model.params.index:
        if not term.startswith(BOROUGH_PREFIX):
            continue
        borough = term[len(BOROUGH_PREFIX):]
        ci = model.conf_int().loc[term]
        rows.append({
            "borough": borough,
            "premium": float(model.params[term]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "pvalue": float(model.pvalues[term]),
        })
    reference = cfg["borough_dummies"]["reference"]
    rows.append({"borough": reference, "premium": 0.0,
                 "ci_low": 0.0, "ci_high": 0.0, "pvalue": 1.0})
    return pd.DataFrame(rows).sort_values("premium")


def plot_premiums(table: pd.DataFrame, out: Path) -> None:
    """Forest plot of the per-borough premiums (95% HC3 CIs)."""
    ypos = list(range(len(table)))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(table["premium"], ypos,
                xerr=[table["premium"] - table["ci_low"],
                      table["ci_high"] - table["premium"]],
                fmt="o", color="black", ecolor="black", capsize=4, ms=5)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(table["borough"])
    ax.set_xlabel("dollar premium vs Manhattan (95% HC3 CI)")
    ax.set_title("Geography premium — dropoff borough (manhattan_sample)")
    for i, row in table.iterrows():
        ax.annotate(f"${row['premium']:+.2f}", xy=(row["premium"], ypos[list(table.index).index(i)]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    df = load_manhattan_sample(cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)

    col = cfg["borough_dummies"]["column"]
    df[col] = df[col].fillna(cfg["geography_premium"]["missing_label"])

    base = fit_baseline(df, cfg)
    geo = fit_geo(df, cfg)
    table = premium_table(geo, cfg)

    before = diagnostics(base)
    after = diagnostics(geo)
    print(f"\n{'metric':<10}{'without':>12}{'with':>12}")
    for m in DIAGNOSTIC_METRICS:
        print(f"{m:<10}{before[m]:>12.4g}{after[m]:>12.4g}")

    print("\n=== per-borough premium vs "
          f"{cfg['borough_dummies']['reference']} (HC3) ===")
    print(table.round(3).to_string(index=False))

    plot_premiums(table, RESULTS / f"{CSV_STEM}.png")
    print(f"wrote {CSV_STEM}.png")

    payload = {
        "method": ("HC3 OLS: fare ~ distance + duration + dropoff_borough "
                   "dummies (reference from config); missing boroughs filled "
                   f"with '{cfg['geography_premium']['missing_label']}'"),
        "n": int(len(df)),
        "reference_borough": cfg["borough_dummies"]["reference"],
        "without_borough": before,
        "with_borough": after,
        "deltas": {m: round(float(after[m] - before[m]), 4)
                   for m in DIAGNOSTIC_METRICS},
        "premiums": table.to_dict("records"),
    }

    out = RESULTS / f"{cfg['sample']['name']}.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data["dataset"] = WORKING_DATASET.name
    data["source_script"] = Path(__file__).name
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["target"] = cfg["target"]
    data["geography_premium"] = payload
    out.write_text(json.dumps(data, indent=2))
    print(f"wrote {out}")

    csv = RESULTS / f"{CSV_STEM}.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")

    diag_csv = RESULTS / f"{CSV_STEM}_diagnostics.csv"
    pd.DataFrame([
        {"metric": m, "without_borough": before[m],
         "with_borough": after[m], "delta": payload["deltas"][m]}
        for m in DIAGNOSTIC_METRICS
    ]).to_csv(diag_csv, index=False)
    print(f"wrote {diag_csv}")


if __name__ == "__main__":
    main()
