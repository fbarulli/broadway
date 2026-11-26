"""04: which forecast model beats the seasonal-naive baseline on the hourly demand series?

Consumes the 01 hourly count CSV (complete buckets only — the partial terminal
hour is dropped, as 02/03 do). The last 14 days are a chronological holdout:
every model forecasts them in one shot and never sees them in training.

Models: persistence (last train value repeated), seasonal-naive-24h (last 24
train hours tiled), train mean, Prophet (daily + weekly seasonality), GluonTS
torch DeepAR, GluonTS torch WaveNet (the CNN stand-in for MQCNN, which GluonTS
>= 0.16 no longer ships in torch), zero-shot TimesFM, statsmodels SARIMAX,
statsforecast AutoARIMA, and a quantile-LightGBM model. The ARIMA and
quantile-LightGBM rows also carry a 95% interval; coverage and width are
validated with the platform calibration metrics on the same holdout.

Every model is a MULTI-STEP forecast of the whole horizon: the naive baselines
are tiled from the train tail (never read from inside the holdout), so the
ranking compares like with like. Stochastic models (DeepAR, WaveNet) are
retrained on REPEATS seeds; the rest are deterministic single runs.

Outputs (all under results/timeseries/):
- 04_forecast_models_metrics.csv: per-run long form model, seed, metric, value
- 04_forecast_models_aggregate.csv: model, metric, runs, mean, std
- 04_forecast_models_intervals.csv: model, alpha, nominal, coverage, mean width
- 04_forecast_models.md: holdout window, ranking, interval validation, verdict
- 04_forecast_models.png: actual vs each model (seed mean), last 7 holdout days
- 04_forecast_models_<metric>.png: one ranked leaderboard per metric, top 3 highlighted
- 04_forecast_models_decision.json: computed verdict vs the seasonal-naive baseline
- 04_forecast_models_champion.json: the champion run with its 95% interval
"""

import importlib
import importlib.metadata
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import SEED, load_decision_rule

from broadway.calibration.metrics import coverage, mean_width
from broadway.calibration.quantile import QuantileInterval
from broadway.evaluate.leaderboard import ChampionDecision, MetricDirection, MetricLeaderboard
from broadway.evaluate.metrics import MAPE_PERCENT_SCALE, compute_metrics
from broadway.reports.decision import Evidence, SegmentEvidence
from broadway.reports.measurement import Measurement, MeasurementEstimator, MeasurementSpec
from broadway.reports.step_output import StepOutput
from broadway.stats.forecast import (
    ArimaOrder,
    AutoArimaForecaster,
    ForecastResult,
    SarimaxForecaster,
    SeasonalOrder,
)
from broadway.training.models.lightgbm import QuantileLGBMRegressor
from broadway.training.models.registry import get_model

RESULTS = Path(__file__).resolve().parent.parent / "results" / "timeseries"
HOURLY_CSV = RESULTS / "01_series_profile_hourly_counts.csv"
STEM = "04_forecast_models"
BLOCK = "day"
REPEATS = 3
SEEDS = tuple(range(REPEATS))
DETERMINISTIC_SEED = -1
HOLDOUT_HOURS = 24 * 14
PLOT_HOURS = 24 * 7
DEEPAR_EPOCHS = 3
WAVENET_EPOCHS = 2
DEEPAR_SAMPLES = 100
TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MAX_CONTEXT = 1024
TIMESFM_MAX_HORIZON = 384  # >= HOLDOUT_HOURS and a multiple of the 128-point output patch
INSTALL_HINT = "bash scripts/uv.sh sync --extra forecast"

INTERVAL_ALPHA = 0.05
SARIMA_ORDER = ArimaOrder(p=1, d=0, q=1)
SARIMA_SEASONAL_ORDER = SeasonalOrder(p=0, d=1, q=1, s=24)
AUTO_ARIMA_SEASON_LENGTH = 24
WEEKEND_DAYS = (5, 6)  # pandas dayofweek: Monday=0 .. Sunday=6

# Which end of each reported metric is better — the leaderboard never infers it.
FORECAST_DIRECTIONS: dict[str, MetricDirection] = {
    "MAE": MetricDirection.MINIMIZE,
    "RMSE": MetricDirection.MINIMIZE,
    "MAPE": MetricDirection.MINIMIZE,
}


def _import_module(module: str, package: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"step 04 needs {package}; install the forecast extra:\n  {INSTALL_HINT}"
        ) from exc


def _collect(skipped: list[str], label: str, build: Callable[[], Any]) -> Any | None:
    """Build an optional model, recording the reason when its package is absent."""
    try:
        return build()
    except ImportError as exc:
        skipped.append(f"`{label}` skipped — {exc}")
        print(f"skip {label}: {exc}")
        return None


def _load_series() -> pd.Series:
    frame = pd.read_csv(HOURLY_CSV, parse_dates=["pickup_datetime"]).sort_values("pickup_datetime")
    complete = frame.set_index("pickup_datetime")["trips"].iloc[:-1]
    # Wall-clock timestamps skip the DST spring-forward hour (2024-03-10 02:00).
    # Interpolate that single hole so lags and GluonTS see one contiguous series;
    # it falls in train, never in the holdout.
    grid = pd.date_range(complete.index[0], complete.index[-1], freq="h")
    return complete.reindex(grid).interpolate(method="linear")


def _baseline_forecasts(train: pd.Series, horizon: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Multi-step naive baselines: the train tail tiled across the horizon.

    Reading the holdout series' own lags (``shift(-1)``) would hand the naive
    baseline the answers; tiling the last train value / last 24 train hours is
    the honest multi-step counterpart of the learned forecasts.
    """
    season = train.to_numpy(dtype=float)[-24:]
    return {
        "persistence": pd.Series(float(train.iloc[-1]), index=horizon),
        "seasonal_naive_24h": pd.Series(np.resize(season, len(horizon)), index=horizon),
        "train_mean": pd.Series(float(train.mean()), index=horizon),
    }


def _forecast_prophet(train: pd.Series, horizon: pd.DatetimeIndex) -> pd.Series:
    prophet = _import_module("prophet", "prophet")
    model = prophet.Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False
    )
    model.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
    forecast = model.predict(pd.DataFrame({"ds": horizon}))["yhat"]
    return pd.Series(forecast.to_numpy(), index=horizon)


def _forecast_timesfm(train: pd.Series, horizon: pd.DatetimeIndex) -> pd.Series:
    timesfm = _import_module("timesfm", "timesfm[torch]")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_CHECKPOINT, torch_compile=False)
    model.compile(
        timesfm.ForecastConfig(
            max_context=TIMESFM_MAX_CONTEXT,
            max_horizon=TIMESFM_MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    context = np.asarray(train.to_numpy(), dtype=np.float32)
    point, _ = model.forecast(horizon=len(horizon), inputs=[context])
    return pd.Series(point[0][: len(horizon)], index=horizon)


def _forecast_sarima(train: pd.Series, horizon: pd.DatetimeIndex) -> ForecastResult:
    return SarimaxForecaster(
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        alpha=INTERVAL_ALPHA,
    ).forecast(train.to_numpy(), len(horizon))


def _forecast_auto_arima(train: pd.Series, horizon: pd.DatetimeIndex) -> ForecastResult:
    return AutoArimaForecaster(
        alpha=INTERVAL_ALPHA, season_length=AUTO_ARIMA_SEASON_LENGTH
    ).forecast(train.to_numpy(), len(horizon))


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar covariates available for the whole forecast horizon."""
    return pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "is_weekend": index.dayofweek.isin(WEEKEND_DAYS).astype(int),
        },
        index=index,
    )


def _forecast_lgbm_quantile(
    train: pd.Series, horizon: pd.DatetimeIndex
) -> ForecastResult:
    model: QuantileLGBMRegressor = get_model("lgbm_quantile", alpha=INTERVAL_ALPHA)
    model.fit(_calendar_features(train.index), train.to_numpy())
    lower, mid, upper = model.predict_interval(_calendar_features(horizon))
    ordered_lower, ordered_upper = QuantileInterval().interval(lower, mid, upper)
    return ForecastResult(
        alpha=INTERVAL_ALPHA,
        point=mid.tolist(),
        lower=ordered_lower.tolist(),
        upper=ordered_upper.tolist(),
    )


def _fit_gluonts(
    estimator_cls: Any,
    epochs: int,
    extra: dict[str, Any],
    dataset: Any,
    horizon: pd.DatetimeIndex,
    seed: int,
) -> pd.Series:
    lightning = _import_module("pytorch_lightning", "gluonts[torch]")
    lightning.seed_everything(seed, workers=True)
    with tempfile.TemporaryDirectory(prefix="broadway-forecast-") as root:
        estimator = estimator_cls(
            freq="h",
            prediction_length=len(horizon),
            trainer_kwargs={
                "max_epochs": epochs,
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "deterministic": True,
                "default_root_dir": root,
            },
            **extra,
        )
        predictor = estimator.train(dataset, num_workers=0)
        forecast = next(iter(predictor.predict(dataset, num_samples=DEEPAR_SAMPLES)))
    return pd.Series(forecast.mean, index=horizon)


def _gluonts_forecasts(
    train: pd.Series, horizon: pd.DatetimeIndex
) -> dict[str, list[tuple[int, pd.Series]]]:
    deepar = _import_module("gluonts.torch.model.deepar", "gluonts[torch]")
    wavenet = _import_module("gluonts.torch.model.wavenet", "gluonts[torch]")
    common = _import_module("gluonts.dataset.common", "gluonts[torch]")
    dataset = common.ListDataset(
        [{"start": pd.Period(train.index[0], freq="h"), "target": train.to_numpy()}], freq="h"
    )
    specs: tuple[tuple[str, Any, int, dict[str, Any]], ...] = (
        ("deepar", deepar.DeepAREstimator, DEEPAR_EPOCHS, {}),
        ("gluonts_wavenet", wavenet.WaveNetEstimator, WAVENET_EPOCHS, {"seasonality": 24}),
    )
    return {
        name: [(seed, _fit_gluonts(cls, epochs, extra, dataset, horizon, seed)) for seed in SEEDS]
        for name, cls, epochs, extra in specs
    }


def _deterministic_forecasts(
    train: pd.Series,
    horizon: pd.DatetimeIndex,
    skipped: list[str],
) -> tuple[dict[str, list[tuple[int, pd.Series]]], dict[str, ForecastResult]]:
    """Point paths for every deterministic model, plus the interval rows."""
    forecasts = _baseline_forecasts(train, horizon)
    intervals: dict[str, ForecastResult] = {}
    optional: dict[str, Callable[[], Any]] = {
        "prophet": lambda: _forecast_prophet(train, horizon),
        "timesfm": lambda: _forecast_timesfm(train, horizon),
        "sarima": lambda: _forecast_sarima(train, horizon),
        "auto_arima": lambda: _forecast_auto_arima(train, horizon),
        "lgbm_quantile": lambda: _forecast_lgbm_quantile(train, horizon),
    }
    for name, build in optional.items():
        predicted = _collect(skipped, name, build)
        if predicted is None:
            continue
        if isinstance(predicted, ForecastResult):
            intervals[name] = predicted
            forecasts[name] = pd.Series(predicted.point, index=horizon)
        else:
            forecasts[name] = predicted
    runs = {name: [(DETERMINISTIC_SEED, path)] for name, path in forecasts.items()}
    return runs, intervals


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE/RMSE/MAPE via the platform metrics owner (MAPE scaled to percent)."""
    values = compute_metrics(
        actual.to_numpy(dtype=float),
        predicted.to_numpy(dtype=float),
        metrics=("mae", "rmse", "mape"),
    )
    return {
        "MAE": values["mae"],
        "RMSE": values["rmse"],
        "MAPE": values["mape"] * MAPE_PERCENT_SCALE,
    }


def _metrics_frame(actual: pd.Series, runs: dict[str, list[tuple[int, pd.Series]]]) -> pd.DataFrame:
    rows = [
        {"model": model, "seed": seed, "metric": metric, "value": round(value, 4)}
        for model, model_runs in runs.items()
        for seed, predicted in model_runs
        for metric, value in _metrics(actual, predicted).items()
    ]
    return pd.DataFrame(rows, columns=["model", "seed", "metric", "value"])


def _aggregate_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        metrics.groupby(["model", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )
    aggregate[["mean", "std"]] = aggregate[["mean", "std"]].round(4)
    return aggregate[["model", "metric", "runs", "mean", "std"]]


def _interval_frame(actual: pd.Series, intervals: dict[str, ForecastResult]) -> pd.DataFrame:
    """Holdout coverage and width of every 95% interval, via the platform metrics."""
    rows = [
        {
            "model": model,
            "alpha": result.alpha,
            "nominal_coverage": 1.0 - result.alpha,
            "coverage": round(
                coverage(actual.to_numpy(), np.asarray(result.lower), np.asarray(result.upper)),
                4,
            ),
            "mean_width": round(
                mean_width(np.asarray(result.lower), np.asarray(result.upper)), 4
            ),
        }
        for model, result in intervals.items()
    ]
    return pd.DataFrame(
        rows, columns=["model", "alpha", "nominal_coverage", "coverage", "mean_width"]
    )


def _mean_paths(runs: dict[str, list[tuple[int, pd.Series]]]) -> dict[str, pd.Series]:
    return {
        model: (
            model_runs[0][1]
            if len(model_runs) == 1
            else pd.concat([predicted for _, predicted in model_runs], axis=1).mean(axis=1)
        )
        for model, model_runs in runs.items()
    }


def _plot_holdout(actual: pd.Series, paths: dict[str, pd.Series]) -> Any:
    window = actual.iloc[-PLOT_HOURS:]
    subplots: Any = plt.subplots(figsize=(15, 6), constrained_layout=True)
    figure, axis = subplots
    axis.plot(window.index, window.to_numpy(), color="black", linewidth=1.8, label="actual")
    for model, predicted in paths.items():
        axis.plot(
            window.index, predicted.reindex(window.index).to_numpy(), linewidth=1.0, label=model
        )
    axis.set(
        title="Holdout forecast — last 7 of the 14 holdout days (seed mean for stochastic)",
        xlabel="",
        ylabel="trips",
    )
    axis.legend(ncol=4, fontsize=8)
    return figure


def _metric_cell(mean: float, std: float) -> str:
    return f"{mean:.1f}" if pd.isna(std) else f"{mean:.1f} ± {std:.1f}"


def _mean_wide(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Model × metric mean — one row per run, the leaderboard's input shape."""
    return aggregate.pivot(index="model", columns="metric", values="mean")


def _write_leaderboards(stem: str, aggregate: pd.DataFrame) -> MetricLeaderboard:
    """One ranked figure per metric; prints the top 3 so the verdict is visible."""
    board = MetricLeaderboard.from_frame(
        _mean_wide(aggregate).reset_index(), id_column="model", metrics=FORECAST_DIRECTIONS
    )
    for metric, path in board.write(RESULTS, stem).items():
        print(f"leaderboard {metric}: top 3 {', '.join(board.top(metric))} -> {path.name}")
    return board


def _mae_interval(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    """95% day-block interval of one model's holdout MAE."""
    frame = pd.DataFrame(
        {
            "actual": actual.to_numpy(dtype=float),
            "predicted": predicted.reindex(actual.index).to_numpy(dtype=float),
            "day": actual.index.floor("D"),
        }
    )
    spec = MeasurementSpec(
        bootstrap="block", block_column=BLOCK, iterations=200, seed=SEED
    )
    measurement = MeasurementEstimator().estimate(
        frame,
        lambda part: float((part["actual"] - part["predicted"]).abs().mean()),
        spec,
        name="mae",
    )
    return measurement.interval_low, measurement.interval_high


def _champion(
    board: MetricLeaderboard, actual: pd.Series, paths: dict[str, pd.Series]
) -> ChampionDecision:
    intervals = {model: _mae_interval(actual, path) for model, path in paths.items()}
    return board.champion("MAE", intervals)


def _improvement(
    actual: pd.Series, champion: pd.Series, naive: pd.Series
) -> tuple[float, tuple[float, float], int, float | None, str]:
    """Champion MAE improvement over seasonal-naive, day-blocked, with its null."""
    frame = pd.DataFrame(
        {
            "actual": actual.to_numpy(dtype=float),
            "champion": champion.reindex(actual.index).to_numpy(dtype=float),
            "naive": naive.reindex(actual.index).to_numpy(dtype=float),
            "day": actual.index.floor("D"),
        }
    )

    def statistic(part: pd.DataFrame) -> float:
        champion_mae = float((part["actual"] - part["champion"]).abs().mean())
        naive_mae = float((part["actual"] - part["naive"]).abs().mean())
        return 1.0 - champion_mae / naive_mae

    spec = MeasurementSpec(
        bootstrap="block",
        block_column=BLOCK,
        null="block_shuffle",
        null_column="actual",
        iterations=200,
        seed=SEED,
    )
    measurement = MeasurementEstimator().estimate(
        frame, statistic, spec, name="mae_improvement", unit="relative_improvement"
    )
    return (
        measurement.value,
        (measurement.interval_low, measurement.interval_high),
        measurement.sample_size,
        measurement.null_threshold,
        measurement.resampling,
    )


def _report_table(wide: pd.DataFrame, spread: pd.DataFrame, runs: pd.Series) -> list[str]:
    rows = []
    for name in wide.index:
        count = int(runs[name])
        kind = "deterministic (single)" if count == 1 else f"stochastic ({count} seeds)"
        cells = " | ".join(
            _metric_cell(wide.loc[name, metric], spread.loc[name, metric])
            for metric in ("MAE", "RMSE", "MAPE")
        )
        rows.append(f"| {name} | {kind} | {cells} |")
    return rows


def _interval_section(intervals: pd.DataFrame) -> list[str]:
    if intervals.empty:
        return []
    rows = [
        f"| {row.model} | {row.nominal_coverage:.2f} | {row.coverage:.3f} | {row.mean_width:.1f} |"
        for row in intervals.itertuples()
    ]
    return [
        "",
        "| interval model | nominal | holdout coverage | mean width (trips) |",
        "|---|---|---|---|",
        *rows,
    ]


def _write_report(
    train: pd.Series,
    actual: pd.Series,
    aggregate: pd.DataFrame,
    intervals: pd.DataFrame,
    skipped: list[str],
    champion: ChampionDecision,
) -> None:
    wide = _mean_wide(aggregate).sort_values("MAE")
    spread = aggregate.pivot(index="model", columns="metric", values="std")
    runs = aggregate.groupby("model")["runs"].max()
    best = champion.champion
    gain = float(wide.loc["persistence", "MAE"]) - float(wide.loc[best, "MAE"])
    lines = [
        "# Forecast model comparison — hourly demand",
        "",
        f"- train: {len(train)} complete hours, {train.index[0]} .. {train.index[-1]}",
        f"- holdout: {len(actual)} hours, {actual.index[0]} .. {actual.index[-1]} (last 14 days)",
        (
            "- preprocessing: partial terminal hour dropped; the DST spring-forward hole "
            "(2024-03-10 02:00) is linearly interpolated for contiguous lags — train only."
        ),
        (
            "- baseline convention: EVERY model is a multi-step forecast of the 336-hour "
            "horizon. persistence tiles the last train value; seasonal-naive-24h tiles the "
            "last 24 train hours. Neither reads the holdout, so the ranking is like-for-like."
        ),
        (
            f"- repeatability: DeepAR and WaveNet are stochastic and retrained on seeds {SEEDS}; "
            "std is across seeds. Every other model is a deterministic single run."
        ),
        (
            "- intervals: SARIMAX (`get_forecast(...).conf_int`), AutoARIMA (statsforecast "
            f"native level) and quantile LightGBM (2.5/50/97.5%) report a {100 * (1 - INTERVAL_ALPHA):.0f}% "
            "holdout interval; coverage/width come from the platform calibration metrics."
        ),
        (
            f"- verdict: champion {best} on MAE = {champion.value:,.1f} trips "
            f"[{champion.interval_low:,.1f}, {champion.interval_high:,.1f}] 95%; "
            f"{gain:+.1f} trips versus persistence, runner-up {champion.runner_up}."
        ),
        "",
        "| model | repeats | MAE | RMSE | MAPE % |",
        "|---|---|---|---|---|",
        *_report_table(wide, spread, runs),
        *_interval_section(intervals),
        "",
        (
            "- context: 02 found the hour-of-day/weekday shape and 03 the 24h/168h ACF dominance "
            "for a series stationary at 5% hourly — hence the 24h naive lag and daily+weekly "
            "seasonality in the learned models (`02_seasonality.md`, `03_stationarity_acf.md`)."
        ),
        "- plot: stochastic models are drawn as their seed-mean path.",
        *(("", "- skipped: " + "; ".join(skipped)) if skipped else ()),
        "",
        (
            f"Artifacts: {STEM}_metrics.csv, {STEM}_aggregate.csv, {STEM}_intervals.csv, "
            f"{STEM}.md, {STEM}.png, one {STEM}_<metric>.png leaderboard per metric, "
            f"{STEM}_champion.json, {STEM}_decision.json (this directory)."
        ),
    ]
    (RESULTS / f"{STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    series = _load_series()
    train, actual = series.iloc[:-HOLDOUT_HOURS], series.iloc[-HOLDOUT_HOURS:]
    print(f"train {len(train)}h -> holdout {len(actual)}h {actual.index[0]} .. {actual.index[-1]}")

    skipped: list[str] = []
    runs, intervals = _deterministic_forecasts(train, actual.index, skipped)
    gluonts = _collect(skipped, "gluonts", lambda: _gluonts_forecasts(train, actual.index))
    if gluonts is not None:
        runs.update(gluonts)

    metrics = _metrics_frame(actual, runs)
    aggregate = _aggregate_frame(metrics)
    interval_frame = _interval_frame(actual, intervals)
    metrics.to_csv(RESULTS / f"{STEM}_metrics.csv", index=False)
    aggregate.to_csv(RESULTS / f"{STEM}_aggregate.csv", index=False)
    interval_frame.to_csv(RESULTS / f"{STEM}_intervals.csv", index=False)

    paths = _mean_paths(runs)
    figure = _plot_holdout(actual, paths)
    board = _write_leaderboards(STEM, aggregate)
    champion = _champion(board, actual, paths)
    (RESULTS / f"{STEM}_champion.json").write_text(
        champion.model_dump_json(indent=2), encoding="utf-8"
    )

    naive_name = "seasonal_naive_24h"
    value, (low, high), n, null_threshold, resampling = _improvement(
        actual, paths[champion.champion], paths[naive_name]
    )
    weekend = actual.index.dayofweek.isin(WEEKEND_DAYS)
    improvement_weekend = 1.0 - float(
        (actual[weekend] - paths[champion.champion].reindex(actual.index)[weekend]).abs().mean()
    ) / float((actual[weekend] - paths[naive_name].reindex(actual.index)[weekend]).abs().mean())
    improvement_weekday = 1.0 - float(
        (actual[~weekend] - paths[champion.champion].reindex(actual.index)[~weekend]).abs().mean()
    ) / float((actual[~weekend] - paths[naive_name].reindex(actual.index)[~weekend]).abs().mean())
    rule = load_decision_rule(STEM)
    evidence = Evidence(
        baseline=f"{naive_name} multi-step forecast (last 24 train hours tiled over the horizon)",
        segments=[
            SegmentEvidence(
                segment="weekday",
                value=improvement_weekday,
                sample_size=int((~weekend).sum()),
            ),
            SegmentEvidence(
                segment="weekend",
                value=improvement_weekend,
                sample_size=int(weekend.sum()),
            ),
        ],
        sensitivity={
            "runner_up_mae": float(
                _mean_wide(aggregate).loc[str(champion.runner_up), "MAE"]
            ),
            "last_7_days_improvement": _improvement(
                actual.iloc[-24 * 7:],
                paths[champion.champion].iloc[-24 * 7:],
                paths[naive_name].iloc[-24 * 7:],
            )[0],
        },
        falsifier="the champion-vs-naive improvement interval falls below the 2% floor",
        provenance={
            "artifact": champion.champion,
            "source": str(HOURLY_CSV),
            "seed": str(SEED),
            "analysis_sample": str(n),
            "block": BLOCK,
            "resampling": resampling,
            "exclusions": "partial terminal hour dropped; DST hole interpolated in train only",
        },
    )
    measured = Measurement(
        statistic=rule.metric,
        value=value,
        interval_low=low,
        interval_high=high,
        null_threshold=null_threshold,
        sample_size=n,
        seed=SEED,
        resampling=resampling,
        block_column=BLOCK,
        iterations=200,
        confidence=0.95,
        unit="relative_improvement",
    )
    decision = measured.to_decision(
        rule,
        question="Does the champion beat the seasonal-naive baseline on the holdout?",
        evidence=evidence,
        reference=None,
    )
    output = StepOutput(
        results_dir=RESULTS, stem=STEM, decision=decision, source=Path(__file__)
    )
    output.add_table("metrics", metrics)
    output.add_table("aggregate", aggregate)
    output.add_table("intervals", interval_frame)
    output.save_figure(figure)
    plt.close("all")
    decision_path = output.write_decision()
    _write_report(train, actual, aggregate, interval_frame, skipped, champion)
    print(aggregate[aggregate["metric"] == "MAE"].sort_values("mean").to_string(index=False))
    print(interval_frame.to_string(index=False))
    print(
        f"champion: {champion.champion} MAE {champion.value:.1f} "
        f"[{champion.interval_low:.1f}, {champion.interval_high:.1f}] "
        f"(runner-up {champion.runner_up}, margin {champion.margin:+.1f})"
    )
    print(
        f"decision: {decision_path} -> {decision.verdict} "
        f"({rule.metric} {value:.4f} [{low:.4f}, {high:.4f}], "
        f"n={n}, null={null_threshold})"
    )


if __name__ == "__main__":
    main()
