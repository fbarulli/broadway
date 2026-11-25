"""04: chronological holdout model comparison on the hourly demand series.

Consumes the 01 hourly count CSV (complete buckets only — the partial terminal
hour is dropped, as 02/03 do). The last 14 days are a chronological holdout:
every model forecasts them in one shot and never sees them in training.

Models: persistence (t-1), seasonal-naive-24h (t-24), train mean, Prophet
(daily + weekly seasonality), GluonTS torch DeepAR, GluonTS torch WaveNet (the
CNN stand-in for MQCNN, which GluonTS >= 0.16 no longer ships in torch), and
zero-shot TimesFM.
Stochastic models (DeepAR, WaveNet) are retrained on REPEATS seeds; the rest
are deterministic single runs. Metrics are reported per seed and as mean ± std.

Outputs (all under results/timeseries/):
- 04_forecast_models_metrics.csv: per-run long form model, seed, metric, value
- 04_forecast_models_aggregate.csv: model, metric, runs, mean, std
- 04_forecast_models.md: holdout window, ranking, verdict, method notes
- 04_forecast_models.png: actual vs each model (seed mean), last 7 holdout days
"""

import importlib
import importlib.metadata
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results" / "timeseries"
HOURLY_CSV = RESULTS / "01_series_profile_hourly_counts.csv"
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


def _import_module(module: str, package: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"step 04 needs {package}; install the forecast extra:\n  {INSTALL_HINT}"
        ) from exc


def _load_series() -> pd.Series:
    frame = pd.read_csv(HOURLY_CSV, parse_dates=["pickup_datetime"]).sort_values("pickup_datetime")
    complete = frame.set_index("pickup_datetime")["trips"].iloc[:-1]
    # Wall-clock timestamps skip the DST spring-forward hour (2024-03-10 02:00).
    # Interpolate that single hole so lags and GluonTS see one contiguous series;
    # it falls in train, never in the holdout.
    grid = pd.date_range(complete.index[0], complete.index[-1], freq="h")
    return complete.reindex(grid).interpolate(method="linear")


def _baseline_forecasts(
    train: pd.Series, series: pd.Series, horizon: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    lags = {"persistence": 1, "seasonal_naive_24h": 24}
    forecasts = {name: series.shift(lag).reindex(horizon) for name, lag in lags.items()}
    forecasts["train_mean"] = pd.Series(float(train.mean()), index=horizon)
    return forecasts


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
    train: pd.Series, series: pd.Series, horizon: pd.DatetimeIndex
) -> dict[str, list[tuple[int, pd.Series]]]:
    forecasts = _baseline_forecasts(train, series, horizon)
    forecasts["prophet"] = _forecast_prophet(train, horizon)
    forecasts["timesfm"] = _forecast_timesfm(train, horizon)
    return {model: [(DETERMINISTIC_SEED, predicted)] for model, predicted in forecasts.items()}


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = actual - predicted
    positive = actual > 0
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "MAPE": float((error[positive].abs() / actual[positive]).mean() * 100.0),
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


def _mean_paths(runs: dict[str, list[tuple[int, pd.Series]]]) -> dict[str, pd.Series]:
    return {
        model: (
            model_runs[0][1]
            if len(model_runs) == 1
            else pd.concat([predicted for _, predicted in model_runs], axis=1).mean(axis=1)
        )
        for model, model_runs in runs.items()
    }


def _plot_holdout(stem: str, actual: pd.Series, paths: dict[str, pd.Series]) -> None:
    window = actual.iloc[-PLOT_HOURS:]
    # plt.subplots is untyped (partially unknown): pin the boundary to Any once.
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
    figure.savefig(RESULTS / f"{stem}.png", dpi=160)
    plt.close(figure)


def _metric_cell(mean: float, std: float) -> str:
    return f"{mean:.1f}" if pd.isna(std) else f"{mean:.1f} ± {std:.1f}"


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


def _write_report(stem: str, train: pd.Series, actual: pd.Series, aggregate: pd.DataFrame) -> None:
    wide = aggregate.pivot(index="model", columns="metric", values="mean").sort_values("MAE")
    spread = aggregate.pivot(index="model", columns="metric", values="std")
    runs = aggregate.groupby("model")["runs"].max()
    best = str(wide.index[0])
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
            "- baseline convention: persistence/seasonal-naive are one-step lags of the observed "
            "series (t-1, t-24); the learned models forecast all 336 hours in one shot."
        ),
        (
            f"- repeatability: DeepAR and WaveNet are stochastic and retrained on seeds {SEEDS}; "
            "std is across seeds. Persistence, seasonal-naive, train-mean, Prophet and "
            "zero-shot TimesFM are deterministic single runs, so their std is blank."
        ),
        f"- verdict: {best} wins on mean MAE: {gain:+.1f} trips versus persistence.",
        "",
        "| model | repeats | MAE | RMSE | MAPE % |",
        "|---|---|---|---|---|",
        *_report_table(wide, spread, runs),
        "",
        (
            "- context: 02 found the hour-of-day/weekday shape and 03 the 24h/168h ACF dominance "
            "for a series stationary at 5% hourly — hence the 24h naive lag and daily+weekly "
            "seasonality in the learned models (`02_seasonality.md`, `03_stationarity_acf.md`)."
        ),
        (
            f"- TimesFM: zero-shot `{TIMESFM_CHECKPOINT}`, package "
            f"`timesfm=={importlib.metadata.version('timesfm')}` (3.0's torch API ships in the "
            "separate unpublished `timesfm3` package, so 2.5 is the latest loadable checkpoint)."
        ),
        (
            "- GluonTS: `gluonts_mqcnn` is unavailable in GluonTS >= 0.16 (MQCNN removed with "
            "the MXNet API); `gluonts_wavenet` is the torch CNN used as the second model."
        ),
        "- plot: stochastic models are drawn as their seed-mean path.",
        "",
        (
            f"Artifacts: {stem}_metrics.csv, {stem}_aggregate.csv, {stem}.md, "
            f"{stem}.png (this directory)."
        ),
    ]
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]))
    print(f"wrote {stem}_metrics.csv, {stem}_aggregate.csv, {stem}.md, {stem}.png")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem
    series = _load_series()
    train, actual = series.iloc[:-HOLDOUT_HOURS], series.iloc[-HOLDOUT_HOURS:]
    print(f"train {len(train)}h -> holdout {len(actual)}h {actual.index[0]} .. {actual.index[-1]}")

    runs = _deterministic_forecasts(train, series, actual.index)
    runs.update(_gluonts_forecasts(train, actual.index))

    metrics = _metrics_frame(actual, runs)
    aggregate = _aggregate_frame(metrics)
    metrics.to_csv(RESULTS / f"{stem}_metrics.csv", index=False)
    aggregate.to_csv(RESULTS / f"{stem}_aggregate.csv", index=False)
    _plot_holdout(stem, actual, _mean_paths(runs))
    _write_report(stem, train, actual, aggregate)
    print(aggregate[aggregate["metric"] == "MAE"].sort_values("mean").to_string(index=False))


if __name__ == "__main__":
    main()
