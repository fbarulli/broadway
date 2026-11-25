"""03: stationarity (ADF) and autocorrelation (ACF) per aggregate (read-only).

Consumes the 01 hourly/daily count CSVs (complete buckets only — the partial
terminal hour/day from 01 is dropped, never patched). No reload of the
canonical frame: this step answers whether the demand series are stationary
and which lags carry memory, the precondition for any forecast baseline.
Outputs (all under results/timeseries/):
- 03_stationarity_adf.csv: ADF stat/p-value/stationary@5% per series
- 03_stationarity_acf_hourly.csv: ACF lags 0-168 (one week) for hourly trips
- 03_stationarity_acf_daily.csv: ACF lags 0-30 for daily trips
- 03_stationarity_acf.md: verdicts plus dominant lags
- 03_stationarity_acf.png: hourly/daily ACF bars with 95% bands
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller

RESULTS = Path(__file__).resolve().parent.parent / "results" / "timeseries"

HOURLY_CSV = RESULTS / "01_series_profile_hourly_counts.csv"
DAILY_CSV = RESULTS / "01_series_profile_daily_counts.csv"
MAX_HOURLY_LAG = 24 * 7
MAX_DAILY_LAG = 30


def _adf_row(name: str, values: pd.Series) -> dict[str, object]:
    stat, p_value, *_ = adfuller(values)
    return {
        "series": name,
        "n": len(values),
        "adf_stat": round(float(stat), 3),
        "p_value": f"{float(p_value):.3g}",
        "stationary_at_5pct": bool(float(p_value) < 0.05),
    }


def main() -> None:
    stem = Path(__file__).stem
    hourly = pd.read_csv(HOURLY_CSV, parse_dates=["pickup_datetime"])
    daily = pd.read_csv(DAILY_CSV, parse_dates=["pickup_datetime"])
    # 01 retains the incomplete terminal buckets in the CSVs: drop them here.
    hourly = hourly.iloc[:-1].reset_index(drop=True)
    daily = daily.iloc[:-1].reset_index(drop=True)

    adf = pd.DataFrame(
        [_adf_row("hourly", hourly["trips"]), _adf_row("daily", daily["trips"])]
    )
    adf.to_csv(RESULTS / f"{stem}_adf.csv", index=False)

    hourly_acf = acf(hourly["trips"], nlags=MAX_HOURLY_LAG, fft=True)
    daily_acf = acf(daily["trips"], nlags=MAX_DAILY_LAG, fft=True)
    pd.DataFrame({"lag": range(len(hourly_acf)), "acf": hourly_acf}).to_csv(
        RESULTS / f"{stem}_acf_hourly.csv", index=False
    )
    pd.DataFrame({"lag": range(len(daily_acf)), "acf": daily_acf}).to_csv(
        RESULTS / f"{stem}_acf_daily.csv", index=False
    )

    # Dominant memory: strongest positive ACF beyond the first seasonal cycle.
    hourly_peaks = sorted(
        ((lag, float(hourly_acf[lag])) for lag in range(24, len(hourly_acf))),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]
    daily_peaks = sorted(
        ((lag, float(daily_acf[lag])) for lag in range(1, len(daily_acf))),
        key=lambda pair: pair[1],
        reverse=True,
    )[:3]

    # plt.subplots is untyped (partially unknown): pin the boundary to Any once.
    subplots: Any = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    figure, axes = subplots
    for axis, values, max_lag, unit in (
        (axes[0], hourly_acf, MAX_HOURLY_LAG, "hours"),
        (axes[1], daily_acf, MAX_DAILY_LAG, "days"),
    ):
        axis.stem(list(range(len(values))), values, linefmt="#2563eb", markerfmt=" ")
        band = 1.96 / (len(values) ** 0.5)
        axis.axhspan(-band, band, color="grey", alpha=0.2)
        axis.set(title=f"ACF of trips ({unit})", xlabel="lag", ylabel="autocorrelation")
    figure.savefig(RESULTS / f"{stem}.png", dpi=160)
    plt.close(figure)

    lines = [
        "# Stationarity + autocorrelation",
        "",
        (
            f"- hourly ADF: stat {adf.loc[0, 'adf_stat']}, p {adf.loc[0, 'p_value']} "
            f"({'stationary' if adf.loc[0, 'stationary_at_5pct'] else 'NON-stationary'} @5%)"
        ),
        (
            f"- daily ADF: stat {adf.loc[1, 'adf_stat']}, p {adf.loc[1, 'p_value']} "
            f"({'stationary' if adf.loc[1, 'stationary_at_5pct'] else 'NON-stationary'} @5%)"
        ),
        (
            "- top hourly lags (24+): "
            + ", ".join(f"{lag}h ({value:.2f})" for lag, value in hourly_peaks)
        ),
        (
            "- top daily lags (1+): "
            + ", ".join(f"{lag}d ({value:.2f})" for lag, value in daily_peaks)
        ),
        "",
        (
            f"Artifacts: {stem}_adf.csv, {stem}_acf_hourly.csv, {stem}_acf_daily.csv, "
            f"{stem}.png (this directory)."
        ),
    ]
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:7]))
    print(f"wrote {stem}_adf.csv, {stem}_acf_hourly.csv, {stem}_acf_daily.csv, {stem}.md, {stem}.png")


if __name__ == "__main__":
    main()
