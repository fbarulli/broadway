"""03: where does the series' memory live — is lag 24 strong enough to beat noise?

Consumes the 01 hourly/daily count CSVs (complete buckets only — the partial
terminal hour/day from 01 is dropped, never patched). The canonical frame is
read once, two columns, only for the CCF second series: hourly mean trip
duration. Answers where the memory lives (ACF/PACF) and how demand leads or
lags duration (CCF) — the precondition for any forecast baseline.

Outputs (all under results/timeseries/):
- 03_stationarity_acf_adf.csv: ADF stat/p-value/stationary@5% per series
- 03_stationarity_acf_acf_hourly.csv: ACF lags 0-168 (one week) for hourly trips
- 03_stationarity_acf_acf_daily.csv: ACF lags 0-30 for daily trips
- 03_stationarity_acf_pacf_hourly.csv: PACF lags 0-168 for hourly trips
- 03_stationarity_acf_pacf_daily.csv: PACF lags 0-30 for daily trips
- 03_stationarity_acf_ccf.csv: CCF of hourly demand vs mean trip duration,
  lags -168..168; positive lag = demand leads duration by that many hours
- 03_stationarity_acf.md: verdicts, dominant lags, significant PACF lags, CCF peak
- 03_stationarity_acf.png: ACF and PACF, hourly and daily, with 95% bands
- 03_stationarity_acf_ccf.png: CCF with 95% band
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from _common import DATETIME_COL, RESULTS, SEED, load_canonical, load_decision_rule

from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.stats.time_series import ADFResult, acf, adf, cross_correlation, pacf

HOURLY_CSV = RESULTS / "01_series_profile_hourly_counts.csv"
DAILY_CSV = RESULTS / "01_series_profile_daily_counts.csv"
DURATION_COL = "trip_duration_minutes"
MAX_HOURLY_LAG = 24 * 7
MAX_DAILY_LAG = 30
MAX_CCF_LAG = 24 * 7
STEM = "03_stationarity_acf"
BLOCK = "day"
LAG = 24


def _acf_at_lag(frame: pd.DataFrame) -> float:
    """Autocorrelation of hourly trips at the daily lag (the baseline's memory)."""
    return float(acf(frame["trips"], LAG)[LAG])


def _is_stationary(result: ADFResult) -> bool:
    """ADF rejects the unit root when the statistic falls below the 5% critical value."""
    return result.statistic < result.critical_values["5%"]


def _adf_row(name: str, values: pd.Series) -> dict[str, object]:
    result = adf(values)
    return {
        "series": name,
        "n": len(values),
        "adf_stat": round(result.statistic, 3),
        "p_value": f"{result.p_value:.3g}",
        "stationary_at_5pct": _is_stationary(result),
    }


def _peak_lags(values: Any, first: int, count: int = 3) -> list[tuple[int, float]]:
    """Strongest positive autocorrelations beyond a starting lag."""
    return sorted(
        ((lag, float(values[lag])) for lag in range(first, len(values))),
        key=lambda pair: pair[1],
        reverse=True,
    )[:count]


def _significant_lags(values: Any, n: int, first: int, count: int = 3) -> list[tuple[int, float]]:
    """PACF lags outside the 95% band, strongest first (the sharp memory)."""
    band = 1.96 / (n ** 0.5)
    return sorted(
        ((lag, float(values[lag])) for lag in range(first, len(values)) if abs(values[lag]) > band),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:count]


def _ccf_frame(hourly: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """CCF of hourly demand vs hourly mean duration, lags -MAX..MAX."""
    frame = load_canonical(columns=[DATETIME_COL, DURATION_COL])
    frame[DATETIME_COL] = pd.to_datetime(frame[DATETIME_COL])
    duration = (
        frame.assign(hour=frame[DATETIME_COL].dt.floor("h"))
        .groupby("hour")[DURATION_COL]
        .mean()
    )
    demand = hourly.set_index("pickup_datetime")["trips"]
    aligned = pd.concat([demand, duration], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("CCF alignment is empty: no overlapping demand/duration hours")
    forward = cross_correlation(aligned[DURATION_COL], aligned["trips"], MAX_CCF_LAG)
    backward = cross_correlation(aligned["trips"], aligned[DURATION_COL], MAX_CCF_LAG)
    lags = list(range(-MAX_CCF_LAG, 0)) + list(range(MAX_CCF_LAG + 1))
    values = list(backward[1:][::-1]) + list(forward)
    return pd.DataFrame({"lag": lags, "ccf": values}), 1.96 / (len(aligned) ** 0.5)


def _plot_acf_pacf_figure(
    hourly_acf: Any, daily_acf: Any, hourly_pacf: Any, daily_pacf: Any, n_hourly: int, n_daily: int
) -> Any:
    subplots: Any = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    figure, axes = subplots
    panels = (
        (axes[0][0], hourly_acf, "hours", n_hourly, "ACF"),
        (axes[0][1], daily_acf, "days", n_daily, "ACF"),
        (axes[1][0], hourly_pacf, "hours", n_hourly, "PACF"),
        (axes[1][1], daily_pacf, "days", n_daily, "PACF"),
    )
    for axis, values, unit, n, label in panels:
        axis.stem(list(range(len(values))), values, linefmt="#2563eb", markerfmt=" ")
        band = 1.96 / (n ** 0.5)
        axis.axhspan(-band, band, color="grey", alpha=0.2)
        axis.set(title=f"{label} of trips ({unit})", xlabel="lag", ylabel="correlation")
    return figure


def _plot_ccf(frame: pd.DataFrame, band: float) -> None:
    subplots: Any = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)
    figure, axis = subplots
    axis.stem(frame["lag"], frame["ccf"], linefmt="#0f766e", markerfmt=" ")
    axis.axhspan(-band, band, color="grey", alpha=0.2)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set(
        title="CCF: hourly demand vs mean trip duration (positive lag = demand leads)",
        xlabel="lag (hours)",
        ylabel="cross-correlation",
    )
    figure.savefig(RESULTS / f"{Path(__file__).stem}_ccf.png", dpi=160)
    plt.close(figure)


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

    hourly_acf = acf(hourly["trips"], MAX_HOURLY_LAG)
    daily_acf = acf(daily["trips"], MAX_DAILY_LAG)
    hourly_pacf = pacf(hourly["trips"], MAX_HOURLY_LAG)
    daily_pacf = pacf(daily["trips"], MAX_DAILY_LAG)
    for name, column, values in (
        ("acf_hourly", "acf", hourly_acf),
        ("acf_daily", "acf", daily_acf),
        ("pacf_hourly", "pacf", hourly_pacf),
        ("pacf_daily", "pacf", daily_pacf),
    ):
        pd.DataFrame({"lag": range(len(values)), column: values}).to_csv(
            RESULTS / f"{stem}_{name}.csv", index=False
        )

    ccf_frame, ccf_band = _ccf_frame(hourly)
    ccf_frame.to_csv(RESULTS / f"{stem}_ccf.csv", index=False)

    hourly_peaks = _peak_lags(hourly_acf, 24)
    daily_peaks = _peak_lags(daily_acf, 1)
    hourly_pacf_lags = _significant_lags(hourly_pacf, len(hourly), 1)
    daily_pacf_lags = _significant_lags(daily_pacf, len(daily), 1)
    peak = ccf_frame.loc[ccf_frame["ccf"].abs().idxmax()]

    rule = load_decision_rule(STEM)
    series_frame = hourly.assign(day=hourly["pickup_datetime"].dt.floor("D"))
    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="trips",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        series_frame, _acf_at_lag, spec, name=rule.metric, unit="correlation"
    )
    evidence = Evidence(
        baseline="no memory: autocorrelation at lag 24 is zero",
        segments=[
            SegmentEvidence(
                segment="first_half",
                value=_acf_at_lag(series_frame.iloc[: len(series_frame) // 2]),
                sample_size=len(series_frame) // 2,
            ),
            SegmentEvidence(
                segment="second_half",
                value=_acf_at_lag(series_frame.iloc[len(series_frame) // 2 :]),
                sample_size=len(series_frame) - len(series_frame) // 2,
            ),
        ],
        sensitivity={
            "daily_acf_lag_1": float(daily_acf[1]),
            "hourly_acf_lag_1": float(hourly_acf[1]),
            "hourly_pacf_lag_24": float(hourly_pacf[LAG]),
        },
        falsifier="the lag-24 autocorrelation interval falls below the 0.20 floor",
        provenance={
            "artifact": "01_series_profile_hourly_counts.csv",
            "source": str(HOURLY_CSV),
            "seed": str(measurement.seed),
            "analysis_sample": str(measurement.sample_size),
            "block": BLOCK,
            "resampling": measurement.resampling,
            "exclusions": "partial terminal hour dropped",
        },
    )
    decision = measurement.to_decision(
        rule,
        question="Is the 24h autocorrelation strong enough for a seasonal-naive baseline?",
        evidence=evidence,
    )

    figure = _plot_acf_pacf_figure(
        hourly_acf, daily_acf, hourly_pacf, daily_pacf, len(hourly), len(daily)
    )
    _plot_ccf(ccf_frame, ccf_band)
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )

    for name, frame in (
        ("adf", adf),
        ("acf_hourly", pd.DataFrame({"lag": range(len(hourly_acf)), "acf": hourly_acf})),
        ("acf_daily", pd.DataFrame({"lag": range(len(daily_acf)), "acf": daily_acf})),
        ("pacf_hourly", pd.DataFrame({"lag": range(len(hourly_pacf)), "pacf": hourly_pacf})),
        ("pacf_daily", pd.DataFrame({"lag": range(len(daily_pacf)), "pacf": daily_pacf})),
        ("ccf", ccf_frame),
    ):
        output_table = frame
        if name != "adf":
            output_table = output_table.head(200)
        output.add_table(name, output_table)
    output.save_figure(figure)
    plt.close("all")
    report, plot, decision_path = output.write()
    print(
        f"ADF hourly stat {adf.loc[0, 'adf_stat']} p {adf.loc[0, 'p_value']}; "
        f"ACF lag24 {measurement.value:.3f}"
    )
    print(
        f"top hourly ACF lags {hourly_peaks}; daily {daily_peaks}; "
        f"PACF hourly {hourly_pacf_lags}, daily {daily_pacf_lags}; "
        f"CCF peak {int(peak['lag'])}h ({float(peak['ccf']):+.3f})"
    )
    print(f"report: {report}")
    print(f"plot: {plot}")
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {measurement.value:.4f} "
        f"[{decision.interval_low:.4f}, {decision.interval_high:.4f}], "
        f"n={measurement.sample_size:,}, null={decision.null_threshold:.4f})"
    )
    print(f"wrote {stem}_adf/acf/pacf/ccf CSVs, {stem}.md, {stem}.png, {stem}_ccf.png")


if __name__ == "__main__":
    main()
