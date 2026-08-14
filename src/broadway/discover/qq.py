"""Joint per-feature Q-Q plots for numeric columns.

Renders standardized quantiles of every numeric feature against theoretical
normal quantiles, one trace per feature, capped at 12 traces per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

MAX_TRACES_PER_FIGURE = 12
MAX_PLOT_POINTS = 10000


class QqFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    n_valid: int
    n_excluded: int
    mean: float
    std: float
    figure: str


class QqOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    total_features: int
    excluded_features: int
    excluded_notes: list[str]
    features: list[QqFeature]
    figures: list[str]
    standardization: str = "per-feature z-score"


def _standardized_quantiles(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (vals - vals.mean()) / vals.std()
    osm, osr = stats.probplot(z, dist="norm", fit=False)
    n = osm.size
    if n > MAX_PLOT_POINTS:
        idx = np.linspace(0, n - 1, MAX_PLOT_POINTS).astype(int)
        osm = osm[idx]
        osr = osr[idx]
    return osm, osr


def _plot_traces(traces: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    for name, osm, osr in traces:
        ax.scatter(osm, osr, s=10, label=name)
    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Theoretical quantiles (standard normal)")
    ax.set_ylabel("Sample quantiles (standardized)")
    ax.set_title("Joint per-feature Q-Q plot (per-feature standardization)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_numeric_qq(df: pd.DataFrame, figures_dir: Path, evidence_path: Path) -> QqOverview:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    total_features = len(numeric_cols)

    figures_dir.mkdir(parents=True, exist_ok=True)

    features: list[QqFeature] = []
    excluded_notes: list[str] = []
    traces: list[tuple[str, np.ndarray, np.ndarray]] = []
    figures: list[str] = []

    for feature in numeric_cols:
        arr = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        non_finite_count = int(np.sum(~np.isfinite(arr)))
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            excluded_notes.append(f"{feature}: non-finite")
            features.append(
                QqFeature(
                    feature=feature,
                    n_valid=0,
                    n_excluded=int(arr.size),
                    mean=0.0,
                    std=0.0,
                    figure="",
                )
            )
            continue
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if std == 0:
            excluded_notes.append(f"{feature}: zero variance")
            features.append(
                QqFeature(
                    feature=feature,
                    n_valid=0,
                    n_excluded=int(arr.size),
                    mean=mean,
                    std=std,
                    figure="",
                )
            )
            continue
        osm, osr = _standardized_quantiles(finite)
        traces.append(
            (feature, osm, osr, int(finite.size), non_finite_count, mean, std)
        )

    figure_index = 1
    while traces:
        chunk = traces[:MAX_TRACES_PER_FIGURE]
        traces = traces[MAX_TRACES_PER_FIGURE:]
        figure_name = f"figures/numeric_qq_{figure_index}.png"
        figures.append(figure_name)
        _plot_traces(
            [(name, osm, osr) for name, osm, osr, *_ in chunk],
            figures_dir / f"numeric_qq_{figure_index}.png",
        )
        for feature, _, _, n_valid, n_excluded, mean, std in chunk:
            features.append(
                QqFeature(
                    feature=feature,
                    n_valid=n_valid,
                    n_excluded=n_excluded,
                    mean=mean,
                    std=std,
                    figure=figure_name,
                )
            )
        figure_index += 1

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    overview = QqOverview(
        source_path=str(evidence_path),
        total_features=total_features,
        excluded_features=len(excluded_notes),
        excluded_notes=excluded_notes,
        features=features,
        figures=figures,
    )
    evidence_path.write_text(overview.model_dump_json(indent=2), encoding="utf-8")
    return overview
