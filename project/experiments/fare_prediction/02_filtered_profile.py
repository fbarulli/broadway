"""02: profile the named fare_prediction_1m sample.

The fare/distance/duration policy lives in the sample definition
(project/config/sample/fare_prediction_1m.yaml), applied once at generation; this
step only consumes the validated sample by name.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from _common import RESULTS, SAMPLE_NAME, UNIT_FMT, box_with_marks

from broadway.samples import read_named_sample

COLS = ["fare_amount", "trip_distance", "trip_duration_minutes"]
PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99, 0.999, 1.0]

CSV_OUT = RESULTS / "02_filtered_profile_describe.csv"
PNG_OUT = RESULTS / "02_filtered_profile.png"


def plot_profiles(df: pd.DataFrame, out_path: Path) -> None:
    """One figure, 3 box-with-marks panels on log-y (shared helper)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, col in zip(axes, COLS):
        box_with_marks(ax, df[col], UNIT_FMT[col], f"{col} (N={len(df)})")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = read_named_sample(SAMPLE_NAME)
    print(f"sample: {SAMPLE_NAME}@{sample.provenance['version']}")
    print(f"rows: {sample.provenance['row_count']}")
    print(f"artifact_sha256: {sample.provenance['artifact_sha256']}")

    df = sample.df
    desc = df[COLS].describe(percentiles=PERCENTILES)
    print(desc)
    desc.to_csv(CSV_OUT)
    print(f"wrote {CSV_OUT}")

    plot_profiles(df, PNG_OUT)
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
