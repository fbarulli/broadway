import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from _common import RESULTS, TARGET, load_sample, INDEPENDENT_NUMERIC_FEATURES

AIRPORT_ZONES = {132: "JFK", 138: "LaGuardia", 237: "Newark"}
HIDDEN_PREMIUM_THRESHOLD = 20.0
FLAT_RATE_BAND = (-10.0, -2.0)

CSV_OUT = RESULTS / "07_airport_flat_rate_audit.csv"
PNG_OUT = RESULTS / "07_airport_residuals.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_sample()

    cols = INDEPENDENT_NUMERIC_FEATURES + ["pickup_location_id", TARGET]
    clean = df[cols].dropna()

    is_airport = clean["pickup_location_id"].isin(AIRPORT_ZONES)
    base = clean[~is_airport]
    airports = clean[is_airport].copy()

    # Metered model fit on non-airport trips only
    X = sm.add_constant(base[INDEPENDENT_NUMERIC_FEATURES])
    model = sm.OLS(base[TARGET], X).fit()
    print("Metered model (non-airport trips):")
    print(model.params.round(4).to_string())

    Xa = sm.add_constant(airports[INDEPENDENT_NUMERIC_FEATURES])
    airports["metered_pred"] = model.predict(Xa)
    airports["residual"] = airports[TARGET] - airports["metered_pred"]
    airports["airport"] = airports["pickup_location_id"].map(AIRPORT_ZONES)

    rows = []
    for name, grp in airports.groupby("airport"):
        resid = grp["residual"]
        rows.append({
            "airport": name,
            "n": len(grp),
            "mean_residual": resid.mean(),
            "median_residual": resid.median(),
            "pct_hidden_premium_20plus": (resid > HIDDEN_PREMIUM_THRESHOLD).mean() * 100,
            "pct_flat_rate_discount": resid.between(*FLAT_RATE_BAND).mean() * 100,
            "fare_distance_corr": grp[TARGET].corr(grp["trip_distance"]),
        })
    out = pd.DataFrame(rows).set_index("airport")
    out.to_csv(CSV_OUT)
    print("\n--- Airport audit ---")
    print(out.round(3).to_string())
    print(f"wrote {CSV_OUT}")

    fig, axes = plt.subplots(2, len(out), figsize=(5 * len(out), 8),
                             constrained_layout=True)
    
    for i, (name, grp) in enumerate(airports.groupby("airport")):
        # Row 1: Histogram
        ax_hist = axes[0, i]
        ax_hist.hist(grp["residual"], bins=100, range=(-30, 30), color="#4c72b0", alpha=0.8)
        ax_hist.axvline(0, color="black", lw=0.8)
        ax_hist.axvline(HIDDEN_PREMIUM_THRESHOLD, color="#d62728", ls="--", lw=1.2, label=f"Hidden >${HIDDEN_PREMIUM_THRESHOLD}")
        ax_hist.set_title(f"{name} (n={len(grp):,})")
        ax_hist.set_xlabel("Actual fare - metered prediction ($)")
        ax_hist.set_ylabel("Frequency")
        if i == 0:
            ax_hist.legend()
        
        # Row 2: Q-Q Plot (subsampled for visual clarity and speed)
        ax_qq = axes[1, i]
        resid_arr = grp["residual"].to_numpy()
        if len(resid_arr) > 5000:
            rng = np.random.default_rng(42 + i)
            resid_sample = rng.choice(resid_arr, 5000, replace=False)
        else:
            resid_sample = resid_arr
            
        sm.qqplot(resid_sample, line='s', ax=ax_qq, fit=True)
        ax_qq.set_title(f"{name} Q-Q Plot")
        ax_qq.set_ylabel("Sample Quantiles")
        if i > 0:
            ax_qq.set_ylabel("")
            
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
