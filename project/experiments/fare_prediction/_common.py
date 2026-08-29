"""Shared paths, constants, plot helper, and the temporal-feature recipe.

The sample is declared once in ``project/config/sample/fare_prediction_1m.yaml``
(seed/size/columns/filters/schema). Steps only consume the name — the sample
registry owns paths, filtering, and sampling. The temporal recipe
(``TEMPORAL_FEATURES`` + ``build_temporal_features``) lives here once so
steps 05 and 06 share it without duplication.
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.ticker import FuncFormatter, LogLocator

from broadway.config.schema import DerivedFeature
from broadway.features.builders import build_derived
from broadway.utils import require_keys
from project.paths import load_project_paths

HERE = Path(__file__).resolve().parent
PATHS = load_project_paths()
RESULTS = PATHS.results / HERE.name
SAMPLE_NAME = "fare_prediction_1m"


def _load_experiment_seed() -> int:
    """Shared seed from the project experiment config (YAML SSOT)."""
    path = PATHS.experiment_configs / "fare_prediction.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require_keys(config, ["seed"], path.name)
    return int(config["seed"])


SEED = _load_experiment_seed()

# Pre-trip feature contract: fare prediction uses only information known
# before the trip starts. Post-trip columns are LEAKAGE and are excluded from
# the model feature set (they may still be stored in the prepared parquets).
# The dropoff zone is known at booking time, so its target encoding is a safe
# pre-trip feature.
POST_TRIP_COLS = ["trip_distance", "trip_duration_minutes", "speed_mph"]
SAFE_NUMERIC_FEATURES = [
    "pickup_hour", "pickup_day_of_week", "pickup_month",
    "is_weekend", "is_rush_hour", "is_night",
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
    "route_id_encoded", "dropoff_location_id_encoded",
    "pu_rush_interaction", "pu_night_interaction", "pu_weekend_interaction",
]
SAFE_CATEGORICAL_FEATURES = ["pickup_location_id", "dropoff_location_id"]
SAFE_FEATURES = SAFE_NUMERIC_FEATURES + SAFE_CATEGORICAL_FEATURES

UNIT_FMT = {
    "fare_amount": "${:g}",
    "trip_distance": "{:g} mi",
    "trip_duration_minutes": "{:.0f} min",
}

MARK_LABELS: tuple[tuple[str, float], ...] = (
    ("min", 0.0),
    ("1%", 0.01),
    ("5%", 0.05),
    ("50%", 0.50),
    ("95%", 0.95),
    ("99%", 0.99),
    ("99.9%", 0.999),
    ("max", 1.0),
)


def box_with_marks(
    ax: plt.Axes,
    values: pd.Series,
    unit_fmt: str,
    title: str,
    tail: bool = True,
    counts: bool = True,
) -> None:
    """Box-whisker on log-y with labeled percentile marks and band counts.

    Whiskers span min→max; the mean is marked; min/1/5/50/95/99/99.9%/max are
    drawn as labeled dashed lines (labels left); each band's value count is
    shown on the right; the region above 99% is shaded as the tail.
    """
    sns.boxplot(
        y=values, log_scale=True, color="#4c72b0", whis=[0, 100],
        width=0.35, ax=ax, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "#d62728",
                   "markeredgecolor": "#d62728", "markersize": 5},
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p, fmt=unit_fmt: fmt.format(v))
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda v, _p, fmt=unit_fmt: fmt.format(v))
    )
    ax.tick_params(axis="y", which="minor", labelsize=6)
    thresholds: list[tuple[float, str]] = []
    for label, p in MARK_LABELS:
        y = float(values.min() if p == 0.0 else
                  values.max() if p == 1.0 else values.quantile(p))
        thresholds.append((y, label))
        ax.axhline(y, color="#d62728", linestyle="--", linewidth=0.8)
        ax.text(0.03, y, f"{label} = {unit_fmt.format(y)}",
                transform=ax.get_yaxis_transform(), fontsize=7,
                color="#d62728", va="bottom")
    if tail:
        ax.axhspan(float(values.quantile(0.99)), float(values.max()),
                   alpha=0.08, color="#d62728")
    if counts:
        for (lo, _), (hi, _) in itertools.pairwise(thresholds):
            n = int(((values > lo) & (values <= hi)).sum())
            ax.text(0.97, (lo * hi) ** 0.5, f"n = {n:,}",
                    transform=ax.get_yaxis_transform(), fontsize=6,
                    color="#555555", va="center", ha="right")
    ax.set_title(title)
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3, axis="y")


RUSH_HOURS = [7, 8, 9, 10, 16, 17, 18, 19]
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
DATETIME_SRC = "pickup_datetime"

TEMPORAL_FEATURES: list[DerivedFeature] = [
    DerivedFeature(name="pickup_hour", func="pickup_hour", source=DATETIME_SRC),
    DerivedFeature(name="pickup_day_of_week", func="pickup_day_of_week", source=DATETIME_SRC),
    DerivedFeature(name="pickup_month", func="pickup_month", source=DATETIME_SRC),
    DerivedFeature(name="is_weekend", func="is_weekend", source=DATETIME_SRC),
    DerivedFeature(name="is_rush_hour", func="is_rush_hour", source=DATETIME_SRC),
    DerivedFeature(name="is_night", func="is_night", source=DATETIME_SRC),
    DerivedFeature(name="hour_sin", func="hour_sin", source=DATETIME_SRC),
    DerivedFeature(name="hour_cos", func="hour_cos", source=DATETIME_SRC),
    DerivedFeature(name="dayofweek_sin", func="dayofweek_sin", source=DATETIME_SRC),
    DerivedFeature(name="dayofweek_cos", func="dayofweek_cos", source=DATETIME_SRC),
]


def _cyclical(values: pd.Series, period: int, trig: str) -> pd.Series:
    """Map ``values`` onto one ``period`` cycle as sin or cos ([-1, 1])."""
    angle = 2.0 * np.pi * values / period
    return np.sin(angle) if trig == "sin" else np.cos(angle)


def _is_rush_hour(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """1 on weekdays whose pickup hour falls in a peak window (7-10 or 16-19)."""
    dt = pd.to_datetime(df[source])
    return ((dt.dt.dayofweek < 5) & (dt.dt.hour.isin(RUSH_HOURS))).astype(int)


def _is_night(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """1 when the pickup hour is 20-23 or 0-5 (hour >= 20 or < 6)."""
    return pd.to_datetime(df[source]).dt.hour.isin(NIGHT_HOURS).astype(int)


def _hour_sin(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Sine of pickup hour on a 24-hour cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.hour, 24, "sin")


def _hour_cos(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Cosine of pickup hour on a 24-hour cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.hour, 24, "cos")


def _dayofweek_sin(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Sine of pickup day-of-week on a 7-day cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.dayofweek, 7, "sin")


def _dayofweek_cos(df: pd.DataFrame, source: str, **kwargs: object) -> pd.Series:
    """Cosine of pickup day-of-week on a 7-day cycle ([-1, 1])."""
    return _cyclical(pd.to_datetime(df[source]).dt.dayofweek, 7, "cos")


_CUSTOM_BUILDERS = {
    "is_rush_hour": _is_rush_hour,
    "is_night": _is_night,
    "hour_sin": _hour_sin,
    "hour_cos": _hour_cos,
    "dayofweek_sin": _dayofweek_sin,
    "dayofweek_cos": _dayofweek_cos,
}


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the shared temporal feature columns (flags + cyclical) to ``df``."""
    return build_derived(df, TEMPORAL_FEATURES, "fare_amount", extra_builders=_CUSTOM_BUILDERS)
