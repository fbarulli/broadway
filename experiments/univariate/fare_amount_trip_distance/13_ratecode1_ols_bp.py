"""13: repeat 04 (OLS + Breusch-Pagan) on the RatecodeID == 1 dataset.

Same analysis as 04_filtered_OLS_BP.py, but on the standard-metered-trip
subset built by 11_ratecode1_dataset.py instead of the filtered 50k sample.
"""

from pathlib import Path

from _common import RESULTS, load_working
from _ols_bp import fit_and_plot_ols_bp

OUT = RESULTS / f"{Path(__file__).stem}.png"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_working()

    result = fit_and_plot_ols_bp(
        df,
        "fare_amount ~ trip_distance",
        OUT,
        suptitle="RatecodeID == 1",
    )

    print(f"working rows: {len(df)}")
    print(f"Breusch-Pagan p-value: {result['bp_pval']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
